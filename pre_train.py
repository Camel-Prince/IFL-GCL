import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, to_scipy_sparse_matrix, dropout_adj
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.data import InMemoryDataset
import random
from model import GRACE, Discriminator, drop_feature
import numpy as np
import pdb
from gca_functional import *
from eval import get_all_embeddings
from torch_geometric.transforms import RandomLinkSplit

class Trainer(object):
    def __init__(self, pt_model: torch.nn.Module,
                 optimizer: torch.optim.Adam,
                 dataset: InMemoryDataset, 
                 device: torch.device, 
                 args, **kwargs) -> None:
        self.args = args
        self.device = device
        self.dataset = dataset
        
        if args.model == 'VGAE':
            self.data = self.dataset.train_data
        else:
            if args.llm_emb:
                self.data = self.dataset[0]
            else:
                self.data = self.dataset._data
        
        # if self.args.senario == 'OOD': # 只有NaP的OOD需要
        #     self.domain_size_dict = {}
        #     PT_domain_id = self.data.domain_id[self.data.PT_mask]
        #     for d_id in PT_domain_id:
        #         if d_id.item() in self.domain_size_dict.keys():
        #             self.domain_size_dict[d_id.item()] += 1
        #         else:
        #             self.domain_size_dict[d_id.item()] = 1
        
        self.features = self.data.x.to(self.device).to(dtype=torch.float32)
        self.edge_index = self.data.edge_index.to(self.device)
        self.pt_model = pt_model
        self.optimizer = optimizer
        self.cur_epoch = 0 # start from 0 epoch
        
        # for PU debias
        # if self.args.PU and not self.args.sim_as_score:
        #     self.PU_Debias_Dict = {
        #         "L_mask": kwargs["L_mask"],
        #         "U_mask": kwargs["U_mask"],
        #         "U_Pos_mask": kwargs["U_Pos_mask"],
        #         "U_Pos_num_vec": kwargs["U_Pos_num_vec"],
        #         "c": kwargs["L_mask"].sum()/(kwargs["L_mask"].sum() + kwargs["U_Pos_mask"].sum()), # label-frequenc in Pos samples
        #         "pi": self.args.pi, # Pos class-prior (pi) as a hyper-param
        #         # "pi": (kwargs["L_mask"].sum() + kwargs["U_Pos_mask"].sum()) / kwargs["L_mask"].numel() # pi的统计结果
        #         # 'z_1': kwargs['z_1'],
        #         # 'z_2': kwargs['z_2']
        #     }
        #     ''' 记得验证pi 和 c的结果是否符合预期'''
        if args.tune_hyper_params:
            self.trial_repeat_params = f'trialNum{args.trial_number}' #_repeat{args.repeat_time}'
            
    def mean_inner_distence(self):  
        """
        计算每个domain内两两节点表示的平均欧式距离
        """     
        # the list of idx to split the X_emb by domain (FB)
        split_idx_list = []
        for i, domain_size in enumerate(self.domain_size_dict.values()):
            split_idx_list.append(sum(list(self.domain_size_dict.values())[:i + 1]))
        
        z = get_all_embeddings(self.pt_model, self.dataset, self.device, self.args)
        embeddings = z.detach().cpu().numpy()
        h_domains = np.split(embeddings, split_idx_list, axis=0)
        h_domains = [torch.tensor(h_domain).to(self.device) for h_domain in h_domains]
        
        inner_dis_list = None
        for domain_emb in h_domains:
            from scipy.spatial.distance import pdist
            # 假设有一个簇的数据点坐标集合
            cluster = np.array(domain_emb.cpu())

            # 计算domain内两两节点之间的曼哈顿距离
            distances = np.array(pdist(cluster, 'euclidean'))
            
            if inner_dis_list is None:
                inner_dis_list = distances
            else:
                inner_dis_list = np.concatenate((inner_dis_list, distances))
        return np.mean(inner_dis_list)

    def pdd(self):
        if self.args.dataset == 'Facebook100':
            all_domains = self.args.pre_train_datasets + [self.args.test_dataset]
            num_domains = len(all_domains)
        else:
            num_domains = torch.unique(self.data.domain_id[self.data.PT_mask]).shape[0]
        
        # the list of idx to split the X_emb by domain (FB)
        split_idx_list = []
        for i, domain_size in enumerate(self.domain_size_dict.values()):
            split_idx_list.append(sum(list(self.domain_size_dict.values())[:i + 1]))
        
        z = get_all_embeddings(self.pt_model, self.dataset, self.device, self.args)
        embeddings = z.detach().cpu().numpy()
        h_domains = np.split(embeddings, split_idx_list, axis=0)
        h_domains = [torch.tensor(h_domain).to(self.device) for h_domain in h_domains]
        pdd = 0
        if num_domains > 1:
            for i in range(num_domains):
                for j in range(i + 1, num_domains):
                    pdd += torch.norm(torch.mean(h_domains[i], dim=0) - torch.mean(h_domains[j], dim=0)) ** 2
            pdd = pdd / (num_domains * (num_domains - 1) / 2)
        return pdd
        
    def train(self):
        """
        Pre-train the model
        """
        self.pt_model.train()
        self.optimizer.zero_grad()
        
        """
        Train the model and BP
        """
        if self.args.model == 'GRACE':
            edge_index_1 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_2)[0]
            x_1 = drop_feature(self.features, self.args.drop_feature_rate_1)
            x_2 = drop_feature(self.features, self.args.drop_feature_rate_2)
            "取出Pretrain需要的embedding"
            if self.args.senario == 'OOD': 
                # 从只在PT上预训练转化为在所有ID的数据上预训练
                ID_mask = self.data.PT_mask + self.data.SFT_mask + self.data.SFT_test_mask
                z1 = self.pt_model(x_1,edge_index_1)[ID_mask] 
                z2 = self.pt_model(x_2,edge_index_2)[ID_mask] 
            else: # 普通的表示学习设定下，所有节点都参与预训练
                z1 = self.pt_model(x_1,edge_index_1)
                z2 = self.pt_model(x_2,edge_index_2)
            loss = self.pt_model.loss(z1=z1, z2=z2, cur_epoch=self.cur_epoch, mean=True, batch_size=self.args.batch_size)
            # domain_size_dict=self.domain_size_dict if self.args.rv_neg_ratio != 0 else None, #
            # PU_Debias_Dict = self.PU_Debias_Dict if self.args.PU and not self.args.sim_as_score else None,
        elif self.args.model == 'DGI':
            x = self.features
            idx = np.random.permutation(x.shape[0])
            x_shuffle = x[idx]
            if self.args.senario == 'OOD':
                ID_mask = self.data.PT_mask + self.data.SFT_mask + self.data.SFT_test_mask
                PT_num_nodes = ID_mask.sum().item()
            else: 
                PT_num_nodes = x.shape[0]
            lbls = torch.cat((torch.ones(PT_num_nodes).to(self.device), torch.zeros(PT_num_nodes).to(self.device)))
            logits = self.pt_model(x, x_shuffle, self.edge_index, mask=ID_mask if self.args.senario == 'OOD' else None)
            loss = F.binary_cross_entropy_with_logits(logits, lbls)
        elif self.args.model == 'BGRL':
            edge_index_1 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_2)[0]
            x_1 = drop_feature(self.features, self.args.drop_feature_rate_1)
            x_2 = drop_feature(self.features, self.args.drop_feature_rate_2)
            _, _, loss = self.pt_model(x_1, x_2, edge_index_1, edge_index_2)
        elif self.args.model == 'COSTA':
            edge_index_1 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_edge(self.edge_index, p=self.args.drop_edge_rate_2)[0]
            x_1 = drop_feature(self.features, self.args.drop_feature_rate_1)
            x_2 = drop_feature(self.features, self.args.drop_feature_rate_2)
            if self.args.senario == 'OOD':
                ID_mask = self.data.PT_mask + self.data.SFT_mask + self.data.SFT_test_mask
                z = self.pt_model(self.features, self.edge_index)[ID_mask]
                z1 = self.pt_model(x_1, edge_index_1)[ID_mask]
                z2 = self.pt_model(x_2, edge_index_2)[ID_mask]
            else:
                z = self.pt_model(self.features, self.edge_index)
                z1 = self.pt_model(x_1, edge_index_1)
                z2 = self.pt_model(x_2, edge_index_2)
            
            k = torch.tensor(int(z.shape[0] * self.args.costa_k_ratio))
            p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(self.device)

            z1 = p @ z1
            z2 = p @ z2 
            
            h1, h2 = [self.pt_model.project(x) for x in [z1, z2]]
            loss = self.pt_model.loss(h1, h2)
        elif self.args.model == 'MVGRL':
            x = self.features
            ft_size = x.shape[1]
            adj = self.data.adj
            diff = self.data.diff
            
            all_sample_size = x.shape[0]
            batch_size = 1
            # if self.args.batch_size == 0 or self.args.batch_size is None:
            #     batch_size = 1
            # else:
            #     batch_size = self.args.batch_size
                
            sample_size = all_sample_size // batch_size
            lbl_1 = torch.ones(batch_size, sample_size * 2)
            lbl_2 = torch.zeros(batch_size, sample_size * 2)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf, b_pt_masks = [], [], [], []
            
            for i in idx:
                ba.append(adj[i: i + sample_size, i: i + sample_size])
                bd.append(diff[i: i + sample_size, i: i + sample_size])
                bf.append(x[i: i + sample_size].cpu().numpy())
                if self.args.senario == 'OOD':
                    ID_mask = self.data.PT_mask + self.data.SFT_mask + self.data.SFT_test_mask
                    b_pt_masks.append(ID_mask[i: i + sample_size])
                else:
                    b_pt_masks.append(torch.ones(sample_size, dtype=torch.bool))

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
            b_pt_masks = np.array(b_pt_masks).reshape(batch_size, sample_size)

            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)
            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            bf = bf.to(self.device)
            ba = ba.to(self.device)
            bd = bd.to(self.device)
            shuf_fts = shuf_fts.to(self.device)
            
            b_xent = torch.nn.BCEWithLogitsLoss()
            logits, __, __ = self.pt_model(bf, shuf_fts, ba, bd, False,None, None, None)
            loss = 0
            
            # 分batch计算loss，注意从所有数据中挑出Pre-train的数据用于计算loss
            valid_batch_num = 0 # 有时候msk全为0，导致loss为nan，所以需要计算有效的batch数；
            for i in range(batch_size):
                msk = np.concatenate((b_pt_masks[i], b_pt_masks[i], b_pt_masks[i], b_pt_masks[i]))
                if msk.sum() == 0:
                    continue
                valid_batch_num += 1
                # msk = (torch.ones(logits.shape[1]) == 1) , just for test the code, to see whether the loss is the same;
                loss += b_xent(logits[i][msk], lbl[i][msk].to(self.device))
            loss = loss / valid_batch_num
            
            # loss = b_xent(logits, lbl.to(self.device))
        elif self.args.model == "GCA":
            def drop_edge(idx: int):
                if self.args.drop_scheme == 'uniform':
                    return dropout_edge(self.edge_index, p=self.args.drop_edge_rate_1 if idx == 1 else self.args.drop_edge_rate_2)[0]
                elif self.args.drop_scheme in ['degree', 'evc', 'pr']:
                    drop_weights = self.data.drop_weights.to(self.device)
                    return drop_edge_weighted(self.edge_index, drop_weights, p=self.args.drop_edge_rate_1 if idx == 1 else self.args.drop_edge_rate_2, threshold=0.7)
                else:
                    raise Exception(f'undefined drop scheme: {self.args.drop_scheme}')

            edge_index_1 = drop_edge(1)
            edge_index_2 = drop_edge(2)
            x_1 = drop_feature(self.features, self.args.drop_feature_rate_1)
            x_2 = drop_feature(self.features, self.args.drop_feature_rate_2)

            if self.args.drop_scheme in ['pr', 'degree', 'evc']:
                feature_weights = self.data.feature_weights.to(self.device)
                x_1 = drop_feature_weighted_2(self.features, feature_weights, self.args.drop_feature_rate_1)
                x_2 = drop_feature_weighted_2(self.features, feature_weights, self.args.drop_feature_rate_2)

            "取出Pretrain需要的embedding"
            if self.args.senario == 'OOD': 
                # 从只在PT上预训练转化为在所有ID的数据上预训练
                ID_mask = self.data.PT_mask + self.data.SFT_mask + self.data.SFT_test_mask
                z1 = self.pt_model(x_1,edge_index_1)[ID_mask] 
                z2 = self.pt_model(x_2,edge_index_2)[ID_mask] 
            else: # 普通的表示学习设定下，所有节点都参与预训练
                z1 = self.pt_model(x_1,edge_index_1)
                z2 = self.pt_model(x_2,edge_index_2)
            loss = self.pt_model.loss(z1, z2, cur_epoch=self.cur_epoch, mean=True, batch_size=self.args.batch_size)
        elif self.args.model == "GBT":
            loss = self.pt_model.loss(data=self.data, device=self.device)
        elif self.args.model == "VGAE": 
            z = self.pt_model.encode(self.features, self.edge_index)
            loss = self.pt_model.recon_loss(z, self.data.pos_edge_label_index)
            loss = loss + (1 / self.features.shape[0]) * self.pt_model.kl_loss()
            # loss = self.pt_model.recon_loss(z, )
        loss.backward()
        self.optimizer.step()
        self.cur_epoch += 1
        return loss
