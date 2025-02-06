import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GAE, VGAE
import pdb
from tqdm import tqdm
from matplotlib.pyplot import matshow, savefig, colorbar, close
import copy
import numpy as np
from typing import Dict, Tuple, Union

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def get_activation_func(activation_str):
    return {'relu': F.relu, 'prelu': nn.PReLU()}[activation_str]
def get_base_model(base_model_str):
    return {'GCNConv': GCNConv, 'SAGEConv': SAGEConv}[base_model_str]

"""
GRACE as Baseline
"""
class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.base_model = get_base_model(args.base_model)
        self.in_channels = args.num_feature
        self.out_channels = args.PU_num_dim if args.estimate==1 else args.num_hidden # 当estimate为1时 encoder用于得到表示，并用来估计Ratio，而不是用于预训练

        assert args.num_layers >= 2
        self.k = args.num_layers
        self.conv = [self.base_model(self.in_channels, 2 * self.out_channels)]
        for _ in range(1, self.k-1):
            self.conv.append(self.base_model(2 * self.out_channels, 2 * self.out_channels))
        self.conv.append(self.base_model(2 * self.out_channels, self.out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = get_activation_func(args.activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, args):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = args.tau
        self.num_hidden = args.PU_num_dim if args.estimate==1 else args.num_hidden
        self.num_proj_hidden = args.num_proj_hidden

        self.fc1 = torch.nn.Linear(self.num_hidden, self.num_proj_hidden)
        self.fc2 = torch.nn.Linear(self.num_proj_hidden, self.num_hidden)

        self.args = args

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, cur_epoch: int):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        B = between_sim
        R = refl_sim
        A = B.sum(1)+R.sum(1)-R.diag()
        
        # 产生+ 和不同view的U+ 的loss
        B_loss_matrix = -torch.log( (B / A.unsqueeze(-1)) )
        
        # 产生同view中的U+ 的loss
        R_loss_matrix = -torch.log( (R / A.unsqueeze(-1)) )
        R_loss_matrix_diag = torch.diag_embed(torch.diag(R_loss_matrix)) 
        R_loss_matrix -= R_loss_matrix_diag # 注意同view中对角线为零，即自己和自己不产生loss
        """
        loss_matrix[i,j] = l_ij = ( p(xi|xj)/p(xi) ) /\sum_{}
        """
        if self.args.theroy_view:  
            # from the theroy view of InfoNCE to update the L\U samples and the loss (the 2nd view)
            """
            -log(L) - log(U_Pos)
            warm up + NaP; no special weight; just take U_Pos as L under certain 锚样本
            """
            if cur_epoch >= self.args.start_debias_epoch:
                # 每隔若干轮再重新筛选U+
                if (cur_epoch-self.args.start_debias_epoch) % self.args.update_interval != 0:
                    B_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_loss_matrix_B_W.pt').to(z1.device)
                    R_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_loss_matrix_R_W.pt').to(z1.device)
                else:
                    R_sim_matrix = f(self.sim(z1, z1)).detach()
                    if self.args.norm_sim_matrix == 'line_by_line':
                        R_W = R_sim_matrix / R_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                    elif self.args.norm_sim_matrix == 'global':
                        R_W = ( R_sim_matrix - R_sim_matrix.min() ) / R_sim_matrix.max()
                    else:
                        raise NotImplementedError("No such norm_sim_matrix way")
                    R_W[R_W < self.args.R_W_threshold] = 0
                    # 自己和自己组成的样本对不算在内，即R_W的对角线为零，不应该算在总数内
                    R_W.fill_diagonal_(0)
                    
                    if self.args.beta != 0:
                        R_W = R_W * self.args.beta
                    
                    B_sim_matrix = f(self.sim(z1, z2)).detach()
                    if self.args.norm_sim_matrix == 'line_by_line':
                        B_W = B_sim_matrix / B_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                    elif self.args.norm_sim_matrix == 'global':
                        B_W = ( B_sim_matrix - B_sim_matrix.min() ) / B_sim_matrix.max()
                    else:
                        raise NotImplementedError("No such norm_sim_matrix way")
                    B_W[B_W < self.args.B_W_threshold] = 0
                    
                    if self.args.beta != 0: 
                        B_W = B_W * self.args.beta
                    
                    # 是否要保持原有正样本的权重为1，且不被筛选
                    if self.args.stay_diag_eye == 1:
                        B_W.fill_diagonal_(1)
                    else:
                        pass
                    
                    torch.save(B_W, f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_loss_matrix_B_W.pt')
                    torch.save(R_W, f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_loss_matrix_R_W.pt')
                
                if cur_epoch % 10 == 0:
                    if self.args.llm_emb:
                        torch.save(B_W, f'UPos/UPos_{self.args.dataset}_Llama3.2-1B_{self.args.model}_{cur_epoch}_B_W.pt')
                    else:
                        torch.save(B_W, f'UPos/UPos_{self.args.dataset}_{self.args.model}_{cur_epoch}_B_W.pt')
                
                R_info_nce = torch.masked_select(R_loss_matrix * R_W, R_W>0)
                B_info_nce = torch.masked_select(B_loss_matrix * B_W, B_W>0)
                info_nce = torch.concat((R_info_nce, B_info_nce)).mean()
            else:
                # 原始的infoNCE
                info_nce = torch.masked_select(B_loss_matrix, torch.eye(z1.shape[0]).to(z1.device).to(torch.bool)).mean()  
        else:
            # 原始infoNCE我们的写法
            info_nce = torch.masked_select(B_loss_matrix, torch.eye(z1.shape[0]).to(z1.device).to(torch.bool)).mean()
            # 原始infoNCE的GRACE写法
            # info_nce = -torch.log( B.diag() / (B.sum(1)+ R.sum(1) - R.diag() ) ).mean()        
        close()
        return info_nce

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int, cur_epoch: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        R_W = None
        B_W = None

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            R = f(self.sim(z1[mask], z1))  # [B, N]
            B = f(self.sim(z1[mask], z2))  # [B, N]
            
            A = B.sum(1) + R.sum(1) - R[:, mask].diag()
            
            B_loss_matrix = -torch.log( (B[:, mask] / A.unsqueeze(-1)) )
        
            # 产生同view中的U+ 的loss
            R_loss_matrix = -torch.log( (R[:, mask] / A.unsqueeze(-1)) )
            R_loss_matrix_diag = torch.diag_embed(torch.diag(R_loss_matrix))
            R_loss_matrix -= R_loss_matrix_diag
            
            if self.args.theroy_view:  
                # from the theroy view of InfoNCE to update the L\U samples and the loss
                """
                -log(L) - log(U_Pos)
                warm up + NaP; no special weight; just take U_Pos as L under certain 锚样本
                """
                if cur_epoch >= self.args.start_debias_epoch:
                    # 每隔若干轮再重新筛选U+
                    if (cur_epoch-self.args.start_debias_epoch) % self.args.update_interval != 0: # 用上一次得到的B_W 和 R_W来筛选U+
                        B_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_batch{i}_loss_matrix_B_W.pt').to(z1.device)
                        R_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_batch{i}_loss_matrix_R_W.pt').to(z1.device)
                        pass
                    else: # 更新R_W和B_W
                        R_sim_matrix = f(self.sim(z1[mask], z1)).detach()
                        if self.args.norm_sim_matrix == 'line_by_line':
                            R_W = R_sim_matrix[:, mask] / R_sim_matrix[:, mask].diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                        elif self.args.norm_sim_matrix == 'global':
                            R_W = ( R_sim_matrix[:, mask] - R_sim_matrix.min() ) / R_sim_matrix.max()
                        else:
                            raise NotImplementedError("No such norm_sim_matrix way")
                        R_W[R_W < self.args.R_W_threshold] = 0
                        # 自己和自己组成的样本对不算在内，即R_W的对角线为零，不应该算在总数内
                        R_W.fill_diagonal_(0)
                        
                        B_sim_matrix = f(self.sim(z1[mask], z2)).detach()
                        if self.args.norm_sim_matrix == 'line_by_line':
                            B_W = B_sim_matrix[:, mask] / B_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                        elif self.args.norm_sim_matrix == 'global':
                            B_W = ( B_sim_matrix[:, mask] - B_sim_matrix.min() ) / B_sim_matrix.max()
                        else:
                            raise NotImplementedError("No such norm_sim_matrix way")
                        B_W[B_W < self.args.B_W_threshold] = 0
                        
                        # 是否要保持原有正样本的权重为1，且不被筛选
                        if self.args.stay_diag_eye == 1:
                            B_W.fill_diagonal_(1)
                        else:
                            pass
                        
                        torch.save(B_W, f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_batch{i}_loss_matrix_B_W.pt')
                        torch.save(R_W, f'Theroy_View_Weights/{self.args.dataset}_{self.args.model}_batch{i}_loss_matrix_R_W.pt')
                    
                    R_info_nce = torch.masked_select(R_loss_matrix * R_W, R_W>0)
                    B_info_nce = torch.masked_select(B_loss_matrix * B_W, B_W>0)
                    info_nce = torch.concat((R_info_nce, B_info_nce))
                else:
                    # 原始的infoNCE
                    info_nce = torch.masked_select(B_loss_matrix, torch.eye(mask.shape[0]).to(device).to(torch.bool))  
            else:
                # 原始infoNCE
                info_nce = torch.masked_select(B_loss_matrix, torch.eye(mask.shape[0]).to(device).to(torch.bool))
            losses.append(info_nce)
            
            # 释放资源
            B = None
            B_loss_matrix = None
            
            R = None
            R_loss_matrix = None
        return torch.cat(losses).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, cur_epoch: int, mean: bool = True, batch_size: int = 0):
        """对比学习部分的loss"""
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        # 拼接
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, cur_epoch)
            l2 = self.semi_loss(h2, h1, cur_epoch)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size, cur_epoch=cur_epoch)
            l2 = self.batched_semi_loss(h2, h1, batch_size, cur_epoch=cur_epoch)

        contrastive_loss  = (l1 + l2) * 0.5
        contrastive_loss = contrastive_loss.mean() if mean else contrastive_loss.sum()
        ret = contrastive_loss
        return ret

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


"""
DGI as a baseline
"""
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_1, h_2, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_1)

        sc_1 = torch.squeeze(self.f_k(h_1, c_x))
        sc_2 = torch.squeeze(self.f_k(h_2, c_x))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits

class DGI(nn.Module):
    def __init__(self, encoder: Encoder,args):
        super(DGI, self).__init__()
        self.gcn = encoder
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(args.num_hidden)

    def forward(self, x, x_shuffle, edge_index, mask=None):
        h_1 = self.gcn(x, edge_index)
        if mask is not None:
            h_1 = h_1[mask]
        
        c = self.read(h_1)
        c = self.sigm(c)

        h_2 = self.gcn(x_shuffle, edge_index)
        if mask is not None:
            h_2 = h_2[mask]

        ret = self.disc(c, h_1, h_2)

        return ret

    # Detach the return variables
    def embed(self, x, edge_index, mask=None):
        h_1 = self.gcn(x, edge_index)
        if mask is not None:
            h_1 = h_1[mask]
        c = self.read(h_1)

        return h_1.detach(), c.detach()

"""
BGRL as a baseline
"""
class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class BGRL(nn.Module):

    def __init__(self, encoder: Encoder, args):
        super().__init__()
        self.student_encoder = encoder
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.moving_average_decay, args.num_epochs)
        rep_dim = args.num_hidden
        pred_hid = args.num_proj_hidden
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2):
        v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1)
        v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1)
            v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2)
        
        
        def loss_fn(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)

        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


"""
COSTA as a baseline
"""
class COSTA(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, args):
        super(COSTA, self).__init__()
        self.encoder = encoder
        self.args = args

        self.fc1 = torch.nn.Linear(self.args.num_hidden, self.args.num_proj_hidden)
        self.fc2 = torch.nn.Linear(self.args.num_proj_hidden, self.args.num_hidden)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def loss(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        sim = h1 @ h2.t() / self.args.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

"""
MVGRL as a baseline
"""
class MVGRL_Discriminator(Discriminator):
    def __init__(self, n_h):
        super().__init__(n_h)
    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits

class MVGRL_Encoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        self.fc = nn.Linear(args.num_feature, args.num_hidden, bias=False)
        self.act = nn.PReLU()

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        return self.act(out)

class MVGRL(nn.Module):
    def __init__(self, encoder: MVGRL_Encoder, args):
        """
        encoder is MVGRL_Encoder which is batched
        """
        super(MVGRL, self).__init__()
        self.gcn1 = encoder
        self.gcn2 = copy.deepcopy(encoder)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = MVGRL_Discriminator(args.num_hidden)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


"""
GCA as a baseline
"""
class GCA(torch.nn.Module):
    def __init__(self, encoder: Encoder, args):
        super(GCA, self).__init__()
        self.args = args
        self.encoder: Encoder = encoder
        self.tau: float = args.tau

        self.fc1 = torch.nn.Linear(args.num_hidden, args.num_proj_hidden)
        self.fc2 = torch.nn.Linear(args.num_proj_hidden, args.num_hidden)

        self.num_hidden = args.num_hidden
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, cur_epoch: int):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        B = between_sim
        R = refl_sim
        A = B.sum(1)+R.sum(1)-R.diag()
        
        # 产生+ 和不同view的U+ 的loss
        B_loss_matrix = -torch.log( (B / A.unsqueeze(-1)) )
        
        # 产生同view中的U+ 的loss
        R_loss_matrix = -torch.log( (R / A.unsqueeze(-1)) )
        R_loss_matrix_diag = torch.diag_embed(torch.diag(R_loss_matrix)) 
        R_loss_matrix -= R_loss_matrix_diag # 注意同view中对角线为零，即自己和自己不产生loss
        """
        loss_matrix[i,j] = l_ij = ( p(xi|xj)/p(xi) ) /\sum_{}
        """
        if self.args.theroy_view:  
            
            # from the theroy view of InfoNCE to update the L\U samples and the loss (the 2nd view)
            """
            -log(L) - log(U_Pos)
            warm up + NaP; no special weight; just take U_Pos as L under certain 锚样本
            """
            if cur_epoch >= self.args.start_debias_epoch:
                # 每隔若干轮再重新筛选U+
                if (cur_epoch-self.args.start_debias_epoch) % self.args.update_interval != 0:
                    B_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_loss_matrix_B_W.pt').to(z1.device)
                    R_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_loss_matrix_R_W.pt').to(z1.device)
                else:
                    R_sim_matrix = f(self.sim(z1, z1)).detach()
                    if self.args.norm_sim_matrix == 'line_by_line':
                        R_W = R_sim_matrix / R_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                    elif self.args.norm_sim_matrix == 'global':
                        R_W = ( R_sim_matrix - R_sim_matrix.min() ) / R_sim_matrix.max()
                    else:
                        raise NotImplementedError("No such norm_sim_matrix way")
                    R_W[R_W < self.args.R_W_threshold] = 0
                    # 自己和自己组成的样本对不算在内，即R_W的对角线为零，不应该算在总数内
                    R_W.fill_diagonal_(0)
                    
                    B_sim_matrix = f(self.sim(z1, z2)).detach()
                    if self.args.norm_sim_matrix == 'line_by_line':
                        B_W = B_sim_matrix / B_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                    elif self.args.norm_sim_matrix == 'global':
                        B_W = ( B_sim_matrix - B_sim_matrix.min() ) / B_sim_matrix.max()
                    else:
                        raise NotImplementedError("No such norm_sim_matrix way")
                    B_W[B_W < self.args.B_W_threshold] = 0
                    
                    # 是否要保持原有正样本的权重为1，且不被筛选
                    if self.args.stay_diag_eye == 1:
                        B_W.fill_diagonal_(1)
                    else:
                        pass
                    
                    torch.save(B_W, f'Theroy_View_Weights/{self.args.dataset}_loss_matrix_B_W.pt')
                    torch.save(R_W, f'Theroy_View_Weights/{self.args.dataset}_loss_matrix_R_W.pt')

                if cur_epoch % 10 == 0:
                    if self.args.llm_emb:
                        torch.save(B_W, f'UPos/UPos_{self.args.dataset}_Llama3.2-1B_{self.args.model}_{cur_epoch}_B_W.pt')
                    else:
                        torch.save(B_W, f'UPos/UPos_{self.args.dataset}_{self.args.model}_{cur_epoch}_B_W.pt')    
                    
                R_info_nce = torch.masked_select(R_loss_matrix * R_W, R_W>0)
                B_info_nce = torch.masked_select(B_loss_matrix * B_W, B_W>0)
                info_nce = torch.concat((R_info_nce, B_info_nce)).mean()
            else:
                # 原始的infoNCE
                info_nce = torch.masked_select(B_loss_matrix, torch.eye(z1.shape[0]).to(z1.device).to(torch.bool)).mean()  
        else:
            # 原始infoNCE我们的写法
            info_nce = torch.masked_select(B_loss_matrix, torch.eye(z1.shape[0]).to(z1.device).to(torch.bool)).mean()
            # 原始infoNCE的GRACE写法
            # info_nce = -torch.log( B.diag() / (B.sum(1)+ R.sum(1) - R.diag() ) ).mean()        
        close()
        return info_nce

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int, cur_epoch: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        R_W = None
        B_W = None

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            R = f(self.sim(z1[mask], z1))  # [B, N]
            B = f(self.sim(z1[mask], z2))  # [B, N]
            
            A = B.sum(1) + R.sum(1) - R[:, mask].diag()
            
            B_loss_matrix = -torch.log( (B[:, mask] / A.unsqueeze(-1)) )
        
            # 产生同view中的U+ 的loss
            R_loss_matrix = -torch.log( (R[:, mask] / A.unsqueeze(-1)) )
            R_loss_matrix_diag = torch.diag_embed(torch.diag(R_loss_matrix))
            R_loss_matrix -= R_loss_matrix_diag
            
            if self.args.theroy_view:  
                # from the theroy view of InfoNCE to update the L\U samples and the loss
                """
                -log(L) - log(U_Pos)
                warm up + NaP; no special weight; just take U_Pos as L under certain 锚样本
                """
                if cur_epoch >= self.args.start_debias_epoch:
                    # 每隔若干轮再重新筛选U+
                    if (cur_epoch-self.args.start_debias_epoch) % self.args.update_interval != 0: # 用上一次得到的B_W 和 R_W来筛选U+
                        B_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_batch{i}_loss_matrix_B_W.pt').to(z1.device)
                        R_W = torch.load(f'Theroy_View_Weights/{self.args.dataset}_batch{i}_loss_matrix_R_W.pt').to(z1.device)
                        pass
                    else: # 更新R_W和B_W
                        R_sim_matrix = f(self.sim(z1[mask], z1)).detach()
                        if self.args.norm_sim_matrix == 'line_by_line':
                            R_W = R_sim_matrix[:, mask] / R_sim_matrix[:, mask].diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                        elif self.args.norm_sim_matrix == 'global':
                            R_W = ( R_sim_matrix[:, mask] - R_sim_matrix.min() ) / R_sim_matrix.max()
                        else:
                            raise NotImplementedError("No such norm_sim_matrix way")
                        R_W[R_W < self.args.R_W_threshold] = 0
                        # 自己和自己组成的样本对不算在内，即R_W的对角线为零，不应该算在总数内
                        R_W.fill_diagonal_(0)
                        
                        B_sim_matrix = f(self.sim(z1[mask], z2)).detach()
                        if self.args.norm_sim_matrix == 'line_by_line':
                            B_W = B_sim_matrix[:, mask] / B_sim_matrix.diag().unsqueeze(-1) # W[i, j] /= W[i,i], for all i, j
                        elif self.args.norm_sim_matrix == 'global':
                            B_W = ( B_sim_matrix[:, mask] - B_sim_matrix.min() ) / B_sim_matrix.max()
                        else:
                            raise NotImplementedError("No such norm_sim_matrix way")
                        B_W[B_W < self.args.B_W_threshold] = 0
                        
                        # 是否要保持原有正样本的权重为1，且不被筛选
                        if self.args.stay_diag_eye == 1:
                            B_W.fill_diagonal_(1)
                        else:
                            pass
                        
                        torch.save(B_W, f'Theroy_View_Weights/{self.args.dataset}_batch{i}_loss_matrix_B_W.pt')
                        torch.save(R_W, f'Theroy_View_Weights/{self.args.dataset}_batch{i}_loss_matrix_R_W.pt')
                    
                    R_info_nce = torch.masked_select(R_loss_matrix * R_W, R_W>0)
                    B_info_nce = torch.masked_select(B_loss_matrix * B_W, B_W>0)
                    info_nce = torch.concat((R_info_nce, B_info_nce))
                else:
                    # 原始的infoNCE
                    info_nce = torch.masked_select(B_loss_matrix, torch.eye(mask.shape[0]).to(device).to(torch.bool))  
            else:
                # 原始infoNCE
                info_nce = torch.masked_select(B_loss_matrix, torch.eye(mask.shape[0]).to(device).to(torch.bool))
            losses.append(info_nce)
            
            # 释放资源
            B = None
            B_loss_matrix = None
            
            R = None
            R_loss_matrix = None
        return torch.cat(losses).mean()


    def loss(self, z1: torch.Tensor, z2: torch.Tensor, cur_epoch: int, mean: bool = True, batch_size = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0 or batch_size is None:
            l1 = self.semi_loss(h1, h2, cur_epoch=cur_epoch)
            l2 = self.semi_loss(h2, h1, cur_epoch=cur_epoch)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size, cur_epoch=cur_epoch)
            l2 = self.batched_semi_loss(h2, h1, batch_size, cur_epoch=cur_epoch)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


"""
GBT as a baseline
"""
EPS = 1e-15
def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss

def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))

def augment(data, p_x: float, p_e: float, device):
    x = data.x.to(device)
    num_fts = x.size(-1)

    ei = data.edge_index.to(device)
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return (x_a, ei_a), (x_b, ei_b)

class GBT(torch.nn.Module):
    def __init__(self, encoder: Encoder, args):
        super(GBT, self).__init__()
        self.args = args
        self.encoder = encoder

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)
    
    def loss(self, data, device):
        (x_a, ei_a), (x_b, ei_b) = augment(data=data, p_x=self.args.p_x, p_e=self.args.p_e, device=device)
        if self.args.senario == 'OOD':
            ID_mask = data.PT_mask + data.SFT_mask + data.SFT_test_mask
            z_a = self.encoder(x=x_a, edge_index=ei_a)[ID_mask]
            z_b = self.encoder(x=x_b, edge_index=ei_b)[ID_mask]
        else:
            z_a = self.encoder(x=x_a, edge_index=ei_a)
            z_b = self.encoder(x=x_b, edge_index=ei_b)
        loss = barlow_twins_loss(z_a=z_a, z_b=z_b)
        return loss
"""
VGAE
"""
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GCNConv(args.num_feature, 2 * args.num_hidden)
        self.conv_mu = GCNConv(2 * args.num_hidden, args.num_hidden)
        self.conv_logstd = GCNConv(2 * args.num_hidden, args.num_hidden)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

"""Unused semi-loss for NaP and PU_debias_By_DesratioEstimation"""
# else: # from a samples' view of InfoNCE to update the L/U samples' and the loss
#     if self.args.senario == 'OOD':
#         # 有抛弃的跨域负样本，就需要mask
#         if self.args.rv_neg_ratio != 0 and cur_epoch >= self.args.start_rv_epoch:
#             domain_idx_range = []
#             domain_size_list = list(domain_size_dict.values())
#             for i in range(len(domain_size_list)):
#                 if i == 0:
#                     domain_idx_range.append((0, domain_size_list[i]))
#                 else:
#                     domain_idx_range.append(
#                         (domain_idx_range[i - 1][1], domain_idx_range[i - 1][1] + domain_size_list[i]))
#             mask = torch.zeros_like(refl_sim, dtype=torch.bool).to(refl_sim.device)
#             for i in range(len(domain_idx_range)):
#                 mask[domain_idx_range[i][0]:domain_idx_range[i][1], domain_idx_range[i][0]:domain_idx_range[i][1]] = True
            
#             # 计算mask
#             if self.args.rv_neg_ratio != 1:
#                 zero_indices = torch.nonzero(mask == 0)
#                 above_zero_indices = zero_indices[zero_indices[:, 0] <= zero_indices[:, 1]]
#                 # 从refl_sim中求出mask
#                 cr_neg_refl_sim = refl_sim[above_zero_indices[:, 0], above_zero_indices[:, 1]]
#                 _, sorted_indices = cr_neg_refl_sim.sort(descending=True)
#                 sorted_ab_zero_idx = above_zero_indices[sorted_indices]
#                 split = int(self.args.rv_neg_ratio*len(sorted_ab_zero_idx))
#                 mask[sorted_ab_zero_idx[split:, 0], sorted_ab_zero_idx[split:, 1]] = True
#                 mask = mask + mask.t()
#                 if cur_epoch > self.args.start_rv_epoch and cur_epoch % 10 == 0:
#                     torch.save(sorted_ab_zero_idx[:split, :], f'{cur_epoch}_sorted_idx.pt')
#                     torch.save(torch.mm(F.normalize(z1), F.normalize(z2).t()),  f'{cur_epoch}_B.pt')
#                 # sizes = torch.tensor(list(domain_size_dict.values()))
#                 # in_domain_num = (sizes * sizes.t()).sum().item()
#                 # tqdm.write(f"实际跨域负样本保留率：{(mask.sum().item()-in_domain_num)/(mask.numel() - in_domain_num):.4f}")
            
#             # 把跨域的正样本加回来
#             if self.args.add_pos_pair:
#                 """
#                 mask没有改变对角线 所以 refll_sim.diag() = (refl_sim * mask).diag(), between_sim.diag() = (between_sim * mask).diag()
                
#                 化简后可以得到最新的info_nce的计算公式
#                 info_nce = -torch.log(
#                     (between_sim.diag() + refl_cross_domain_pos_sim.sum(1) + between_cross_domain_pos_sim.sum(1)) /
#                     (refl_sim.sum(1)- refl_sim.diag() + between_sim.sum(1))
#                 )
#                 """
#                 between_cross_domain_pos_sim = between_sim - between_sim * mask
#                 refl_cross_domain_pos_sim = refl_sim - refl_sim * mask
#                 updated_mask = mask.clone()
                
#                 info_nce = -torch.log(
#                     (between_sim.diag() + refl_cross_domain_pos_sim.sum(1) + between_cross_domain_pos_sim.sum(1)) /
#                     (refl_sim.sum(1)- refl_sim.diag() + between_sim.sum(1))
#                 )
                
#             # 只把跨域的负样本抛弃， 不把跨域的正样本加回来
#             else:
#                 info_nce = -torch.log(
#                     between_sim.diag()
#                     / ((refl_sim * mask).sum(1)- refl_sim.diag() + (between_sim * mask).sum(1) )
#                 )
#         else: # 没有抛弃的跨域负样本，也没有添加跨域正样本，就不需要mask
#             info_nce = -torch.log( between_sim.diag() / (refl_sim.sum(1)- refl_sim.diag() + between_sim.sum(1) ) )
#     else: # 普通的表示学习场景（又可分成纠偏和不纠偏的部分）
#         if self.args.PU: # 使用PU纠偏 在PU预处理的部分引入r(L) > r(U)的约束是暂时不可行的，因为r(x)直接收numpy，不能会穿梯度
#             if (self.args.sim_as_score and cur_epoch >= self.args.start_PU_epoch) or not self.args.sim_as_score:
#                 if self.args.sim_as_score and cur_epoch >= self.args.start_PU_epoch: 
#                     # sim_matrix = torch.mm(F.normalize(z1), F.normalize(z2).t()).to(z1.device).detach()
#                     sim_matrix = f(self.sim(z1, z1)).detach()
#                     from utils import get_masks
#                     L_mask, U_mask, U_Pos_mask, U_Pos_num_vec = get_masks(sim_matrix, sim_matrix.device, self.args)
#                     c = L_mask.sum()/(L_mask.sum() + U_Pos_mask.sum())
#                 else: #不使用Sim_Matrix作为Score 即 not self.args.sim_as_score
#                     L_mask = PU_Debias_Dict["L_mask"]
#                     U_mask = PU_Debias_Dict["U_mask"]
#                     U_Pos_mask = PU_Debias_Dict["U_Pos_mask"]
#                     U_Pos_num_vec = PU_Debias_Dict["U_Pos_num_vec"]
#                     c = PU_Debias_Dict['c']
#                 pi = self.args.pi
#                 N = L_mask.shape[0]
                
#                 if self.args.u_pos_loss == 'sum':
#                     debias_Pos = (c * between_sim * L_mask).sum(1) + \
#                                 (1-c) * 1/2 * ( (between_sim*U_Pos_mask).sum(1) + (refl_sim*U_Pos_mask).sum(1))
#                 elif self.args.u_pos_loss == 'mean':
#                     debias_Pos = (c * between_sim * L_mask).sum(1) + \
#                                 (1-c) * 1/2 * ( (between_sim*U_Pos_mask).sum(1) /  U_Pos_num_vec + (refl_sim*U_Pos_mask).sum(1) /  U_Pos_num_vec )
                
#                 import math
#                 debias_Neg = (2*N -2) * \
#                 ( (1-pi*c)/(1-pi) * 1/(2*N-2) * ( (between_sim*U_mask).sum(1) + (refl_sim*U_mask).sum(1) ) - 
#                 pi*(1-c)/(1-pi) * 1/2 * ( (between_sim*U_Pos_mask).sum(1) /  U_Pos_num_vec + (refl_sim*U_Pos_mask).sum(1) /  U_Pos_num_vec )
#                 )
#                 debias_Neg[debias_Neg < 1/math.e ] = 1/math.e
                
                
#                 info_nce = -torch.log(
#                     debias_Pos / (debias_Pos + debias_Neg)
#                 )
#             else: # sim_as_score 但是 还没到start_PU_epoch
#                 info_nce = -torch.log( between_sim.diag() / (refl_sim.sum(1)- refl_sim.diag() + between_sim.sum(1) ) )
#         else: # 不使用PU纠偏
#             info_nce = -torch.log( between_sim.diag() / (refl_sim.sum(1)- refl_sim.diag() + between_sim.sum(1) ) )