import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. Graph Definition (COCO 17-Keypoint Layout)
# -----------------------------------------------------------------------------
class Graph:
    def __init__(self, strategy='uniform'):
        self.num_node = 17
        self.edges = self._get_edges()
        self.center = 0 # Nose or Pelvis depending on layout, 0 is safe for now
        self.A = self._get_adjacency_matrix(strategy)

    def _get_edges(self):
        # Edges for COCO 17-keypoint layout
        # (Start_Node, End_Node) - 0-indexed
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # Face
            (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
            (11, 13), (13, 15), (12, 14), (14, 16), # Legs
            (5, 6), (11, 12),                    # Shoulders/Hips connections
            (5, 11), (6, 12),                    # Trunk (L/R)
            (0, 5), (0, 6)                       # Head to Shoulders (Approx neck)
        ]
        return edges

    def _get_adjacency_matrix(self, strategy):
        # Create plain Adjacency Matrix
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        
        # Normalize
        # A_norm = D^(-1/2) * (A + I) * D^(-1/2)
        A = A + np.eye(self.num_node)
        D = np.sum(A, axis=1)
        D_hat = np.power(D, -0.5)
        A_norm = np.dot(np.dot(np.diag(D_hat), A), np.diag(D_hat))
        
        return torch.tensor(A_norm, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 2. Graph Convolution Layer
# -----------------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # A is fixed (non-learnable structure for simplicity in this project)
        # Registering as buffer means it's part of state_dict but not updated by optimizer
        self.register_buffer('A', A)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x shape: (N, C, T, V)
        N, C, T, V = x.size()
        
        # 1. Graph Convolution: A * x
        # We process each frame independently for the spatial step
        # A is (V, V), x is (N, C, T, V)
        # We want result (N, C, T, V)
        
        # Merge N and T to treat frames as a batch for multiplication
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C) # (NT, V, C)
        
        # Matrix Mul: (V, V) @ (NT, V, C) -> (NT, V, C)
        x_graph = torch.matmul(self.A, x_reshaped) 
        
        # Reshape back: (N, T, V, C) -> (N, C, T, V)
        x_graph = x_graph.view(N, T, V, C).permute(0, 3, 1, 2)
        
        # 2. Feature Transform (1x1 Conv)
        out = self.conv(x_graph)
        out = self.bn(out)
        
        return F.relu(out)

# -----------------------------------------------------------------------------
# 3. ST-GCN Block (Spatial + Temporal)
# -----------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0):
        super(STGCNBlock, self).__init__()
        
        # Spatial Graph Conv
        self.gcn = GraphConv(in_channels, out_channels, A)
        
        # Temporal Conv (Standard 2D Conv over Time axis)
        # Kernel size (9, 1) means looking at 9 frames (with padding), 1 node
        # Since we only have 6 frames, we use a smaller kernel size like (3,1) or (5,1)
        kernel_size = (3, 1) 
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=(stride, 1), 
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

# -----------------------------------------------------------------------------
# 4. Main Model
# -----------------------------------------------------------------------------
class STGCNActionClassifier(nn.Module):
    def __init__(self, num_classes=27, in_channels=3):
        super(STGCNActionClassifier, self).__init__()
        
        graph = Graph()
        self.A = graph.A
        
        # Data Normalization (input batch norm)
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        
        # Network Body
        # We keep it shallow because Sequence Length T=6
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, self.A, stride=1, dropout=0.1),
            STGCNBlock(64, 64, self.A, stride=1, dropout=0.1),
            STGCNBlock(64, 128, self.A, stride=1, dropout=0.1), # Stride=2 would reduce time dim, but T=6 is small so keep 1
            STGCNBlock(128, 256, self.A, stride=1, dropout=0.1),
        ])
        
        # Classification Head
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Input: [Batch, T, V, C] or [Batch, T, 17, 3]
        Output: [Batch, Num_Classes]
        """
        # Permute to [N, C, T, V] which is standard for Conv2d implementation
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # -> [N, 3, 6, 17]
        
        # Normalization
        x = x.view(N, -1, T) # Flatten C*V for BatchNorm1d: [N, 3*17, 6]
        x = self.data_bn(x)
        x = x.view(N, C, T, V)
        
        # Forward pass blocks
        for layer in self.layers:
            x = layer(x)
            
        # Global Pooling over Time (T) and Vertices (V)
        # x is [N, 256, T, V]
        x = F.avg_pool2d(x, x.size()[2:]) # -> [N, 256, 1, 1]
        
        # Prediction
        x = self.fcn(x) # -> [N, Num_Classes, 1, 1]
        x = x.view(x.size(0), -1) # -> [N, Num_Classes]
        
        return x

if __name__ == '__main__':
    # Test
    # Batch=4, Frames=6, Joints=17, Coords=3
    input_skel = torch.rand(4, 6, 17, 3) 
    model = STGCNActionClassifier(num_classes=27)
    output = model(input_skel)
    
    print("Input Shape:", input_skel.shape)
    print("Output Shape:", output.shape) # Expect [4, 27]