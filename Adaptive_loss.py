# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 08:12:26 2023

@author: zxc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:58:02 2023

@author: zxc
"""


import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.models.vgg import vgg16
import numpy as np
import cv2
# from scipy.fftpack import fft2
def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad

# import torchvision.transforms.functional as TF



# class L1_Charbonnier_loss(torch.nn.Module):
#     """L1 Charbonnierloss."""
#     def __init__(self):
#         super(L1_Charbonnier_loss, self).__init__()
#         self.eps = 1e-6

#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt(diff * diff + self.eps)
#         loss = torch.mean(error)
#         return loss
    
# class L2_Charbonnier_loss(torch.nn.Module):
#     """L2 Charbonnier loss."""
#     def __init__(self):
#         super(L2_Charbonnier_loss, self).__init__()
#         self.eps = 1e-6

#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt(diff * diff + self.eps)
#         loss = torch.mean(error * error)
#         return loss
# def gradient(img):
#     dx = img[:, :, :, 1:] - img[:, :, :, :-1]
#     dy = img[:, :, 1:, :] - img[:, :, :-1, :]
#     return dx, dy
# def gradient_loss(pred, target):
#     pred_dx, pred_dy = gradient(pred)
#     target_dx, target_dy = gradient(target)
#     dx_loss = F.mse_loss(pred_dx, target_dx)
#     dy_loss = F.mse_loss(pred_dy, target_dy)
#     return dx_loss + dy_loss

# def LOG(img):
#     [B,C,H,W]=img.shape
#     gaussian_kernel = torch.tensor([
#     [1, 2, 1],
#     [2, 4, 2],
#     [1, 2, 1]
#     ], dtype=torch.float32) / 16
#     gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3)
#     img_gaussian = torch.zeros(B,C,H,W)
#     for i in range(C):
#         img_gaussian[:,i,:,:] = F.conv2d(img[:,i,:,:], gaussian_kernel, padding=1)
#     laplacian_kernel = torch.tensor([
#     [0, 1, 0],
#     [1, -4, 1],
#     [0, 1, 0]
#     ], dtype=torch.float32)
#     laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
#     img_laplacian = torch.zeros(B,C,H,W)
#     for i in range(C):
#         img_laplacian[:,i,:,:] = F.conv2d(img_gaussian[:,i,:,:], laplacian_kernel, padding=1)
    
#     return img_laplacian

# def spectral_gradient(fused, ref):
#     sg_fused = fused[:, 0: fused.size(1)-1, :, :] - fused[:, 1:fused.size(1), :, :]
#     sg_ref =   ref[:, 0:ref.size(1)-1, :, :] - ref[:, 1:ref.size(1), :, :]
    
#     return sg_fused, sg_ref

# class Con_Edge_Spec_loss(nn.Module):
#     def __init__(self):
#         super(Con_Edge_Spec_loss, self).__init__()
#         self.L1_loss = nn.L1Loss()
#         self.L2_loss = nn.MSELoss()
#     def forward(self, fused, ref, alpha=1, beta=1):
        
#         Con_loss = self.L1_loss(fused, ref)
        
       
#         # Edge_loss  = self.L1_loss(edge_fused, edge_ref)
#         Edge_loss  = gradient_loss(fused, ref)
        
#         spec_fused, spec_ref = spectral_gradient(fused, ref)
#         Spec_loss = self.L2_loss(spec_fused, spec_ref)
        
#         loss = Con_loss + Edge_loss + Spec_loss
        
#         # print(loss)
#         return loss 
    
    
class GeneratorLoss(nn.Module):
    def __init__(self, alpha = 1.0):
        super(GeneratorLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        # self.beta = nn.Parameter(torch.tensor(beta, requires_grad=True))
        self.tv_loss = TVLoss()
        self.l1loss = nn.L1Loss()
        
    def forward(self, img_out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - img_out_labels)#+0.3*torch.mean(1 - edge_out_labels)+0.3*torch.mean(1 - spec_out_labels)
       
        image_loss = self.l1loss(out_images, target_images)
       
        tv_loss = self.tv_loss(out_images)
        # return image_loss + adversarial_loss  + 2e-8 * tv_loss
        # return image_loss   + 2e-8 * tv_loss
        return image_loss + self.alpha*adversarial_loss  + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

