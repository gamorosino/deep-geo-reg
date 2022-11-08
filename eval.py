#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system('nvidia-smi')


# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import math
import nibabel as nib
import os
import struct
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from utils import *
get_ipython().run_line_magic('matplotlib', 'inline')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[ ]:


# settings

# data
data_dir = './data/'
cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
D, H, W = 224, 160, 224

# misc
device = 'cuda'

# keypoints / graph
d_fixed = 5
scale_moving = 1.25
d_moving = 3
k = 10
k1 = 256

# displacement space
l_max = 9
l_width = l_max * 2 + 1
q = 3
disp = torch.stack(torch.meshgrid(torch.arange(- q * l_max, q * l_max + 1, q),
                                  torch.arange(- q * l_max, q * l_max + 1, q),
                                  torch.arange(- q * l_max, q * l_max + 1, q))).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float()
disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(device)

# patch
patch_radius = 2
patch_step = 2
patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step),
                                   torch.arange(0, 2 * patch_radius + 1, patch_step),
                                   torch.arange(0, 2 * patch_radius + 1, patch_step))).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float() - patch_radius
patch = (patch.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(device)

# pca
pca_components = 64

#LBPS
slbp_iter = 5
slbp_cost_scale = 10
slbp_alpha = -50

#LBPD
dlbp_iter = 5
dlbp_cost_scale = 1
dlbp_alpha = -15


# In[ ]:


# load data
orig_shapes  = torch.load(os.path.join(data_dir, 'preprocessed/resize_s/orig_shapes.pth'))
imgs_fixed   = torch.zeros(len(cases),  1, D, H, W).pin_memory()
masks_fixed  = torch.zeros(len(cases),  1, D, H, W).pin_memory()
imgs_moving  = torch.zeros(len(cases),  1, D, H, W).pin_memory()
masks_moving = torch.zeros(len(cases),  1, D, H, W).pin_memory()

for i, case in enumerate(cases):
    print('loading case {} ...'.format(case))
    t0 = time.time()

    path_img_fixed   = os.path.join(data_dir, 'preprocessed/resize_s/case{}_img_fixed.nii.gz'.format(case + 1))
    path_mask_fixed  = os.path.join(data_dir, 'preprocessed/resize_s/case{}_mask_fixed.nii.gz'.format(case + 1))
    path_img_moving  = os.path.join(data_dir, 'preprocessed/resize_s/case{}_img_moving.nii.gz'.format(case + 1))
    path_mask_moving = os.path.join(data_dir, 'preprocessed/resize_s/case{}_mask_moving.nii.gz'.format(case + 1))

    img_fixed   = (torch.from_numpy(nib.load(path_img_fixed).get_data()).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500
    mask_fixed  = torch.from_numpy(nib.load(path_mask_fixed).get_data()).unsqueeze(0).unsqueeze(0).bool()
    img_moving  = (torch.from_numpy(nib.load(path_img_moving).get_data()).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500
    mask_moving = torch.from_numpy(nib.load(path_mask_moving).get_data()).unsqueeze(0).unsqueeze(0).bool()

    imgs_fixed[i]   = img_fixed
    masks_fixed[i]  = mask_fixed
    imgs_moving[i]  = img_moving
    masks_moving[i] = mask_moving

    t1 = time.time()

    print('{:.2f} s'.format(t1-t0))


# In[ ]:


def tre(case, disp_pred):
    Do, Ho, Wo = orig_shapes[case].long().tolist()
    path_lms_fixed   = os.path.join(data_dir, 'preprocessed/resize_s/case{}_lms_fixed.nii.gz'.format(case + 1))
    lms_fixed   = kpts_pt(img_to_kpts(torch.from_numpy(nib.load(path_lms_fixed).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device), 300), (D, H, W))
    lms_fixed_disp_pred = F.grid_sample(disp_pred.flip(1), lms_fixed.view(1, -1, 1, 1, 3), padding_mode='border', align_corners=True).view(3,-1).t()
    lms_fixed_disp_pred = lms_fixed_disp_pred / 2 * (torch.tensor([Do, Ho, Wo]).to(device) - 1)
    if case < 10:
        return copdgene_error(os.path.join(data_dir, 'COPDgene'), case, lms_fixed_disp_pred.cpu())
    elif case < 20:
        return  dct4_error(os.path.join(data_dir, '4DCT'), case - 10, lms_fixed_disp_pred.cpu())
    
def discretize(kpts_fixed, kpts_fixed_feat, kpts_moving, kpts_moving_feat):
    N_p_fixed = kpts_fixed.shape[1]
    disp_range = disp.max(1, keepdim=True)[0]
    dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3, keepdim=True)
    grid = inverse_grid_sample(candidates_cost.view(N_p_fixed, 1, -1), candidates[0]/disp_range, (l_width, l_width, l_width), mode='nearest', padding_mode='zeros', align_corners=True)
    grid_norm = inverse_grid_sample(torch.ones_like(candidates_cost.view(N_p_fixed, 1, -1)), candidates[0]/disp_range, (l_width, l_width, l_width), mode='nearest', padding_mode='zeros', align_corners=True)
    cost = grid  / (grid_norm + 0.000001)
    cost[cost==0] = 1e4
    return cost

def lbp_graph(kpts_fixed):
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    return edges, edges_reverse_idx

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x, ind):
        B, N, D = x.shape
        k = ind.shape[2]
        
        y = x.view(B*N, D)[ind.view(B*N, k)].view(B, N, k, D)
        x = x.view(B, N, 1, D).expand(B, N, k, D)
        
        x = torch.cat([y - x, x], dim=3)
        
        x = self.conv(x.permute(0, 3, 1, 2))
        x = F.max_pool2d(x, (1, k))
        x = x.squeeze(3).permute(0, 2, 1)
        
        return x
    
class GeometricFeatNet(nn.Module):
    def __init__(self):
        super(GeometricFeatNet, self).__init__()
        
        self.conv1 = EdgeConv(3, 32)
        self.conv2 = EdgeConv(32, 32)
        self.conv3 = EdgeConv(32, 64)
        
        self.conv4  = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False),
                                    nn.InstanceNorm1d(64),
                                    nn.Conv1d(64, 64, 1))
        
    def forward(self, kpts_fixed, kpts_moving, k):
        fixed_ind = knn_graph(kpts_fixed, k, include_self=True)[0]
        x = self.conv1(kpts_fixed, fixed_ind)
        x = self.conv2(x, fixed_ind)
        x = self.conv3(x, fixed_ind)
        
        moving_ind = knn_graph(kpts_moving, k*3, include_self=True)[0]
        y = self.conv1(kpts_moving, moving_ind)
        y = self.conv2(y, moving_ind)
        y = self.conv3(y, moving_ind)
        
        x = self.conv4(x.permute(0,2,1)).permute(0,2,1)
        y = self.conv4(y.permute(0,2,1)).permute(0,2,1)
        
        return x, y


# In[ ]:


def sLBP(kpts_fixed, kpts_moving):
    N_p_fixed = kpts_fixed.shape[1]

    # inference
    dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = torch.ones(1, N_p_fixed, k1, 1).to(device)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(slbp_iter):
        temp_message = torch.zeros((N_p_fixed, k1)).to(device).scatter_add_(0, edges[:, 1].view(-1, 1).expand(-1, k1), messages)
        multi_data_cost = torch.gather(temp_message + candidates_cost.squeeze(), 0, edges[:,0].view(-1, 1).expand(-1, k1))
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1))
        multi_data_cost -= reverse_messages
        messages = torch.zeros_like(multi_data_cost)
        unroll_factor = 32
        split = np.array_split(np.arange(multi_data_cost.shape[0]), unroll_factor)
        for i in range(unroll_factor):
            messages[split[i]] = torch.min(multi_data_cost[split[i]].unsqueeze(1) + slbp_cost_scale*(candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3), 2)[0]
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    kpts_fixed_disp_pred = (candidates * F.softmax(slbp_alpha * reg_candidates_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)).sum(2)

    return kpts_fixed_disp_pred

def sLBP_GF(kpts_fixed, kpts_moving, net):
    N_p_fixed = kpts_fixed.shape[1]

    # geometric features
    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)

    # inference
    dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(slbp_iter):
        temp_message = torch.zeros((N_p_fixed, k1)).to(device).scatter_add_(0, edges[:, 1].view(-1, 1).expand(-1, k1), messages)
        multi_data_cost = torch.gather(temp_message + candidates_cost.squeeze(), 0, edges[:,0].view(-1, 1).expand(-1, k1))
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1))
        multi_data_cost -= reverse_messages
        messages = torch.zeros_like(multi_data_cost)
        unroll_factor = 32
        split = np.array_split(np.arange(multi_data_cost.shape[0]), unroll_factor)
        for i in range(unroll_factor):
            messages[split[i]] = torch.min(multi_data_cost[split[i]].unsqueeze(1) + slbp_cost_scale*(candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3), 2)[0]
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    kpts_fixed_disp_pred = (candidates * F.softmax(slbp_alpha * reg_candidates_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)).sum(2)
    
    return kpts_fixed_disp_pred

def sLBP_MIND(img_fixed, img_moving, kpts_fixed, kpts_moving):
    N_p_fixed = kpts_fixed.shape[1]

    # MIND featutes
    mind_fixed  = mindssc(img_fixed)
    mind_moving = mindssc(img_moving)

    # sample patches
    kpts_fixed_mind = F.grid_sample(mind_fixed, kpts_fixed.view(1, 1, -1, 1, 3) + patch.view(1, 1, 1, -1, 3), mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1, 4).contiguous().view(1, -1, 12*patch.view(-1,3).shape[0])
    kpts_moving_mind = F.grid_sample(mind_moving, kpts_moving.view(1, 1, -1, 1, 3) + patch.view(1, 1, 1, -1, 3), mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1, 4).contiguous().view(1, -1, 12*patch.view(-1,3).shape[0])

    # PCA feat embedding
    v, mean = pca_train(torch.cat([kpts_fixed_mind, kpts_moving_mind], 1), pca_components)
    kpts_fixed_feat = pca_fit(kpts_fixed_mind, v, mean)
    kpts_moving_feat = pca_fit(kpts_moving_mind, v, mean)

    # inference
    dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(slbp_iter):
        temp_message = torch.zeros((N_p_fixed, k1)).to(device).scatter_add_(0, edges[:, 1].view(-1, 1).expand(-1, k1), messages)
        multi_data_cost = torch.gather(temp_message + candidates_cost.squeeze(), 0, edges[:,0].view(-1, 1).expand(-1, k1))
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1))
        multi_data_cost -= reverse_messages
        messages = torch.zeros_like(multi_data_cost)
        unroll_factor = 32
        split = np.array_split(np.arange(multi_data_cost.shape[0]), unroll_factor)
        for i in range(unroll_factor):
            messages[split[i]] = torch.min(multi_data_cost[split[i]].unsqueeze(1) + slbp_cost_scale*(candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3), 2)[0]
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    kpts_fixed_disp_pred = (candidates * F.softmax(slbp_alpha * reg_candidates_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)).sum(2)

    return kpts_fixed_disp_pred

def dLBP_GF(kpts_fixed, kpts_moving, net):
    N_p_fixed = kpts_fixed.shape[1]

    # geometric features
    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)

    # match
    cost = discretize(kpts_fixed, kpts_fixed_feat, kpts_moving, kpts_moving_feat)
    edges, _ = lbp_graph(kpts_fixed)
    messages = torch.zeros_like(cost)
    for _ in range(dlbp_iter):
        message_data = messages + cost
        reg_message_data = minconv(dlbp_cost_scale*message_data, l_width)/dlbp_cost_scale
        messages = torch.zeros_like(cost).view(N_p_fixed, -1).scatter_add_(0, edges[:, 0].view(-1, 1).expand(-1, l_width**3), reg_message_data[edges[:, 1]].view(-1, l_width**3)).view_as(cost)
        
    reg_cost = messages + cost
    kpts_fixed_disp_pred = (disp.unsqueeze(1) * F.softmax(dlbp_alpha * reg_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)).sum(2)
    
    return kpts_fixed_disp_pred


# In[ ]:


with torch.no_grad():
    methods = ['sLBP', 'sLBP+GF', 'sLBP+MIND', 'dLBP+GF']
    tres = torch.zeros(len(methods), len(cases))
    times = torch.zeros(len(methods), len(cases))
    for i, method in enumerate(methods):
        print('Method: {}'.format(method))
        print()
        for j, case in enumerate(cases):
            print('Case {}'.format(case))
            print('-------')

            # load data and transfer to GPU (asynchronus)
            img_fixed   = imgs_fixed[j:j+1].to(device, non_blocking=True)
            mask_fixed  = masks_fixed[j:j+1].to(device, non_blocking=True)
            img_moving  = imgs_moving[j:j+1].to(device, non_blocking=True)
            mask_moving = masks_moving[j:j+1].to(device, non_blocking=True)
            
            # sample kpts
            kpts_fixed = foerstner_kpts(img_fixed, mask_fixed, d=d_fixed)
            kpts_moving = foerstner_kpts(F.interpolate(img_moving, scale_factor=scale_moving, mode='trilinear', align_corners=True), (F.interpolate(mask_moving, scale_factor=scale_moving, mode='trilinear') > 0), d=d_moving)
            
            # preload models
            if method in ['sLBP+GF', 'dLBP+GF']:
                net = GeometricFeatNet()
                net.load_state_dict(torch.load('./models_slbp+gf/case{}_final.pth'.format(case)))
                net.to(device)
                net.eval()
                
            # methods
            torch.cuda.synchronize()
            t0 = time.time()
            
            if method == 'sLBP':
                kpts_fixed_disp_pred = sLBP(kpts_fixed, kpts_moving)
            if method == 'sLBP+GF':
                kpts_fixed_disp_pred = sLBP_GF(kpts_fixed, kpts_moving, net)
            if method == 'sLBP+MIND':
                kpts_fixed_disp_pred = sLBP_MIND(img_fixed, img_moving, kpts_fixed, kpts_moving)
            if method == 'dLBP+GF':
                kpts_fixed_disp_pred = dLBP_GF(kpts_fixed, kpts_moving, net)
                 
            torch.cuda.synchronize()
            t1 = time.time()

            # densify
            disp_pred = densify(kpts_fixed, kpts_fixed_disp_pred, (D//3, H//3, W//3))
            disp_pred = F.interpolate(disp_pred, mode='trilinear', size=(D, H, W))

            # eval
            tres[i, j] = tre(case, disp_pred).mean()
            print('tre: {} (initial: {})'.format(tres[i, j].item(), tre(case, 0*disp_pred).mean().item()))
            
            # time
            times[i, j] = t1-t0
            print('time(s): {}'.format(t1-t0))
            print()

        print('--------------')
        print('mean tre: {}'.format(tres[i, :].mean().item()))
        print('mean time: {}'.format(times[i, :].mean().item()))
        print()
                


# In[ ]:




