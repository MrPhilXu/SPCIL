import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["AdaSPLoss"]

class AdaSPLoss(object):
    """
    自适应稀疏配对（AdaSP）损失
    """

    def __init__(self, N_id, temp=0.04, device="cuda:0", loss_type='adasp'):
        self.temp = temp
        self.loss_type = loss_type
        self._device = device
        self.N_id = N_id

    def __call__(self, feats, targets):
        # print("----------------adasp loss------------")
        # 归一化输入特征
        feats_n = nn.functional.normalize(feats, dim=1)

        bs_size = feats_n.size(0)
        # bs_size = 128
        # N_id = len(torch.unique(targets))
        N_id = self.N_id
        N_ins = bs_size // N_id
        scale = 1. / self.temp

        # 计算相似性矩阵
        sim_qq = torch.matmul(feats_n, feats_n.T)
        # print("sim_qq.shape: ", sim_qq.shape)
        sf_sim_qq = sim_qq * scale
        # print("sf_sim_qq.shape: ", sf_sim_qq.shape)

        # 准备用于构建掩码的因子
        right_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).to(self._device)
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).to(self._device)
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1, N_ins)))).to(self._device)
        
        ## 正对正挖掘
        # 创建正对正正样本的掩码
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id), -1. * np.ones((N_ins, N_ins)))).to(self._device)
        mask_HH[mask_HH == 0] = 1.

        # 计算正对正样本的相似性得分
        # print("sf_sim_qq.shape: ", sf_sim_qq.shape)
        # print("mask_HH.shape:  ", mask_HH.shape)
        # sim_qq.shape: torch.Size([128, 128])
        # sf_sim_qq.shape: torch.Size([128, 128])
        # sf_sim_qq.shape: torch.Size([128, 128])
        # mask_HH.shape: torch.Size([126, 126])
        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        # 准备正样本的掩码
        pos_mask_id = torch.eye(N_id).to(self._device)
        # 增强正样本的相似性得分
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH == 0] = 1.
        pos_sim_HH = 1. / pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1 - pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
        
        # 归一化相似性得分
        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH, p=1, dim=1)   
        
        ## 正对负挖掘
        # 创建正对负样本的掩码
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id), -1. * np.ones((N_ins, N_ins)))).to(self._device)
        mask_HE[mask_HE == 0] = 1.
        # 计算正对负样本的相似性得分
        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        # 增强正对负样本的相似性得分
        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE == 0] = 1.
        pos_sim_HE = 1. / pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1 - pos_mask) + pos_sim_HE.mul(pos_mask)

        # 计算负对负样本的相似性得分
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        # 归一化相似性得分
        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE, p=1, dim=1)
        
    
        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        # 计算组合不同相似性得分的权重
        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach() / scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach() / scale
        wt_l = 2 * weight_sim_HE.mul(weight_sim_HH) / (weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1 - wt_l) 
    
        # 计算正样本的自适应相似性
        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).to(self._device)
        # 结合自适应相似性和正对负相似性
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1 - pos_mask_id)

        # 归一化自适应相似性得分
        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat, p=1, dim=1)

        # 计算不同类型的损失
        loss_sph = -1 * torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_splh = -1 * torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adasp = -1 * torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
        
        # 根据指定的损失类型选择相应的损失
        if self.loss_type == 'sp-h':
            loss = loss_sph.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_splh.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adasp
            
        return loss
