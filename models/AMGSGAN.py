# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:47:52 2023

@author: zxc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:48:18 2023

@author: zxc
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x



def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class AMGSGAN(nn.Module):

    def __init__(self, scale_ratio, n_select_bands,  n_bands,  feature_dim=256, alpha = 1.0, beta = 1.0, gamma=1.0, mlp_dim=[256, 128]):
        super().__init__()        
        
        self.encoder_HSI = nn.Sequential(
                  nn.Conv2d(n_bands, feature_dim, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.encoder_MSI = nn.Sequential(
                  nn.Conv2d(n_select_bands, feature_dim, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_3_HSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        self.conv_5_HSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_7_HSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=3, dilation=3)
        
        self.conv_3_MSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        self.conv_5_MSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_7_MSI = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=3, dilation=3)
        
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.scale_ratio = scale_ratio
        self.mlp_dim=mlp_dim
        self.feature_dim = feature_dim
        
        
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=True))
        
        imnet_in_dim = self.feature_dim + self.feature_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=n_bands+1, hidden_list=self.mlp_dim)
        
    def query(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w).cuda()

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret


    def forward(self, HSI, MSI):
        HSI_encoder = self.encoder_HSI(HSI)
        MSI_encoder = self.encoder_MSI(MSI)
        HSI_3 = self.conv_3_HSI(HSI_encoder)
        HSI_5 = self.conv_5_HSI(HSI_encoder)
        HSI_7 = self.conv_7_HSI(HSI_encoder)
        
        MSI_3 = self.conv_3_MSI(MSI_encoder)
        MSI_5 = self.conv_5_MSI(MSI_encoder)
        MSI_7 = self.conv_7_MSI(MSI_encoder)
        # print(MSI_3.shape)
        # print(MSI_5.shape)
        _, _, H, W = MSI.shape
        
        coord = make_coord([H, W]).cuda()
        
        A = self.query(HSI_3, coord, MSI_3)
        B = self.query(HSI_5, coord, MSI_5)
        C = self.query(HSI_7, coord, MSI_7)
        
        Out = self.alpha*A + self.beta*B + self.gamma*C
        return Out,0,0,0,0,0
        
        
if __name__ == '__main__':
    
    model = AMGSGAN(scale_ratio=4, n_select_bands=4,  n_bands=103).cuda()
    
    HSI = torch.rand(1,103,32,32).cuda()
    MSI = torch.rand(1,4,128,128).cuda()
    
    T = model(HSI,MSI)
    
    print('T:',T[0].shape)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        