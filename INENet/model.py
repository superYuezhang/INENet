import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math, sys
sys.path.append("..")
from feature_extract import PointNet


use_cuda = torch.cuda.is_available()


class STNkd(nn.Module):
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class OverlapNet(nn.Module):
    def __init__(self, num_subsampled_points, n_emb_dims=1024):
        super(OverlapNet, self).__init__()
        self.stn = STNkd(k=3)
        self.emb_dims = n_emb_dims
        self.emb_nn = PointNet(n_emb_dims=self.emb_dims)
        # 第二个点云的点数(缺失点云的点数)
        self.num_subsampled_points = num_subsampled_points
        self.threshold_nn = nn.Sequential(nn.Linear(self.emb_dims, 512),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.01),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(negative_slope=0.01),
                                nn.Linear(256, 128),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(negative_slope=0.01),
                                nn.Linear(128, 1),
                                nn.Sigmoid())
        self.mask_nn = nn.Sequential(nn.Conv1d(self.num_subsampled_points, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
                                nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
								nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
                                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
								nn.Conv1d(128, 1, 1), nn.Sigmoid())
        self.sigm = nn.Sigmoid()

    def forward(self, *input):
        src = input[0]  # 1024
        tgt = input[1]  # 768
        trans_src = self.stn(src)
        trans_tgt = self.stn(tgt)
        src = torch.bmm(src.transpose(2, 1), trans_src).transpose(2, 1)
        tgt = torch.bmm(tgt.transpose(2, 1), trans_tgt).transpose(2, 1)
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        batch_size, num_dims, num_points1 = src_embedding.size()
        batch_size, num_dims, num_points2 = tgt_embedding.size()

        src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
        tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
        # tar_norm = tgt_glob / (tgt_glob.norm(dim=1).reshape(batch_size, 1, 1))

        cos_simi = torch.matmul(tar_norm.transpose(2, 1).contiguous(),
                              src_norm)  # (batch, num_points2, num_points1)
       
        # ----方法2----网络预测阈值法
        src_glob = torch.mean(src_embedding, dim=2)
        tar_glob = torch.mean(tgt_embedding, dim=2)
        glob_residual = torch.abs(src_glob - tar_glob)
        threshold = self.threshold_nn(glob_residual)
        # threshold = 0.5
        mask = self.mask_nn(cos_simi).reshape(batch_size, -1)
        mask_idx = torch.where(mask >= threshold, 1, 0)
        # ----end----

        return mask, mask_idx, threshold


# # src,tar:[batchsize, 3, num_points]
# src = torch.rand([4, 3, 1024]).cuda()
# tar = torch.rand([4, 3, 768]).cuda()
# model = OverlapNet().cuda()
# mask, mask_idx, threshold = model(src, tar)
# print(mask.dtype, threshold.dtype )


