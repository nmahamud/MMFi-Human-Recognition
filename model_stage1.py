import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Core PointNet++ Operations (Pure PyTorch - No C++ Compilation Needed)
# -----------------------------------------------------------------------------

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src: source points, [B, N, C]
    dst: target points, [B, M, C]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    # Initialize with random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        # Update centroids
        centroids[:, i] = farthest
        
        # Get coordinates of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        # Calculate distance from this centroid to all other points
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        # Update distances (keep min distance to any centroid)
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Pick the point with the max distance
        farthest = torch.max(distance, -1)[1]
        
        # --- SAFETY FIX ---
        # If input points are all the same (e.g. all zeros), 
        # distance might be 0 for all, and torch.max might behave unpredictably or return index 0.
        # We ensure the index is always valid.
        farthest = torch.clamp(farthest, 0, N-1)

    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# -----------------------------------------------------------------------------
# 2. PointNet++ Modules
# -----------------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        """
        xyz = xyz.permute(0, 2, 1) # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1) # [B, N, D]

        if self.group_all:
            new_xyz = xyz[:, :1, :] # Dummy
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.unsqueeze(1).permute(0, 3, 2, 1) # [B, C+D, N, 1]
        else:
            # Farthest Point Sampling
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            
            # Ball Query
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], self.npoint, 1, 3) # Center coords
            
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
            else:
                new_points = grouped_xyz

            new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]

        # MLP Layers
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max Pooling
        new_points = torch.max(new_points, 2)[0] # [B, C_out, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# -----------------------------------------------------------------------------
# 3. Main Stage 1 Model: PointNet++ Regressor
# -----------------------------------------------------------------------------

class PointNet2Regressor(nn.Module):
    def __init__(self, num_joints=17):
        super(PointNet2Regressor, self).__init__()
        
        # 1. Set Abstraction Layers (Downsampling)
        # Input: 512 points. Features: 5 (x,y,z + doppler + intensity)
        # Note: in_channel is 2 (doppler + intensity) because xyz is handled separately in SA
        
        # SA1: 512 -> 128 points
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=2+3, mlp=[64, 64, 128], group_all=False)
        
        # SA2: 128 -> 32 points
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=32, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        
        # SA3: Global Pooling (All -> 1 feature vector)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)
        
        # 2. Regression Head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        # Output: 17 joints * 3 coordinates = 51
        self.fc3 = nn.Linear(256, num_joints * 3)

    def forward(self, xyz):
        """
        Input: [B, N, 5]  (x,y,z, doppler, intensity)
        Output: [B, 17, 3]
        """
        # Split XYZ and Features
        # xyz inputs needs to be [B, 3, N] for the module
        pos = xyz[:, :, :3].permute(0, 2, 1) # [B, 3, N]
        feat = xyz[:, :, 3:].permute(0, 2, 1) # [B, 2, N]

        # Pass through Set Abstraction layers
        l1_xyz, l1_points = self.sa1(pos, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # Global Feature [B, 1024, 1]

        x = l3_points.view(l3_points.size(0), -1) # Flatten [B, 1024]
        
        # Regression Head
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x) # [B, 51]
        
        # Reshape to skeleton format
        x = x.view(-1, 17, 3)
        return x

# -----------------------------------------------------------------------------
# 4. Loss Function (MPJPE)
# -----------------------------------------------------------------------------

class MPJPELoss(nn.Module):
    def __init__(self):
        super(MPJPELoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Mean Per Joint Position Error
        pred: [B, 17, 3]
        target: [B, 17, 3]
        """
        # Euclidian distance per joint
        dist = torch.norm(pred - target, dim=-1) # [B, 17]
        # Mean over joints and batch
        return torch.mean(dist)

if __name__ == '__main__':
    # Test the model with dummy data
    model = PointNet2Regressor()
    input_data = torch.rand(4, 512, 5) # Batch of 4, 512 points, 5 features
    output = model(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output.shape) # Should be [4, 17, 3]