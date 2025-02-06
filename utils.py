from sklearn.manifold import TSNE
import argparse
from model import GRACE
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Dataset
import random
from eval import get_all_embeddings
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def tSNE(features, emb_dim=2, perplexity=30.0):
    tsne = TSNE(n_components=emb_dim, init='pca', learning_rate='auto', perplexity=perplexity)
    features_emb = tsne.fit_transform(features)

    return features_emb

def vis_pt_test_embs(dataset: Dataset, pt_model: torch.nn.Module, args, device: torch.device, **kwargs):
    # 设置全局字体为Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 30
    marker_size = 30
    data = dataset._data
    
    embeddings = get_all_embeddings(pt_model, dataset, device, args)
    labels = data.y.to(device)
    domain_size_dict = data.domain_size_dict

    num_labels = torch.unique(labels).shape[0]
    X_emb = embeddings.detach().cpu().numpy()
    
    if args.dataset == 'Facebook100':
        all_domains = args.pre_train_datasets + [args.test_dataset]
        num_domains = len(all_domains)
    else:
        num_domains = torch.unique(data.domain_id).shape[0]
    
    # the list of idx to split the X_emb by domain (FB)
    split_idx_list = []
    for i, domain_size in enumerate(domain_size_dict.values()):
        split_idx_list.append(sum(list(domain_size_dict.values())[:i + 1]))
    
    # compute PDD before tsne
    h_domains = np.split(embeddings, split_idx_list, axis=0)
    pdd = 0
    if num_domains > 1:
        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                pdd += torch.norm(torch.mean(h_domains[i], dim=0) - torch.mean(h_domains[j], dim=0)) ** 2
        pdd = pdd / (num_domains * (num_domains - 1) / 2)
        
    
    X_emb = tSNE(X_emb)
    X_emb_domains = np.split(X_emb, split_idx_list, axis=0)
    
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(121)
    domain_color_tuples = [('#FF0000', '#FF6666', '#990000'), ('#00FF00', '#66FF66', '#006600'),('#0000FF', '#6699FF', '#000099'), ('#FFFF00', '#FFFF66', '#999900'),('#00FFFF', '#66FFFF', '#009999'), ('#FF00FF', '#FF66FF', '#990099'),('#CCCCCC', '#FFFFFF', '#666666'), ('#FF77AA', '#FFAAEE', '#770033'),('#00CCFF', '#66FFFF', '#006699'), ('#FFCC00', '#FFDD66', '#996600'),('#FF3366', '#FF6688', '#990033'), ('#00FFCC', '#66FFEE', '#009988'),('#FF9900', '#FFBB66', '#994400'), ('#CC00FF', '#EE66FF', '#660099'),('#99FF00', '#BBFF66', '#446600'), ('#FF0033', '#FF3366', '#99001A') ]
   
    for i in range(num_domains):
        X_emb_domain = X_emb_domains[i]
        ax.scatter(X_emb_domain[:, 0], X_emb_domain[:, 1], marker="o", color=domain_color_tuples[i][0],
                   label=all_domains[i] if args.dataset == 'Facebook100' else  f"{args.dataset} domain{i}" , 
                   s=marker_size, alpha=1)
    ax.legend(prop={'size': font_size},loc=3, bbox_to_anchor=(-0.3, 0))
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.05)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.set_title(f'Color By Domain\nPDD: {pdd:.4f}', fontdict={'fontsize': font_size})

    # plot the node-embedding color by cls_id(label)
    cls_color_tuples = [
    (
    "#FF00FF",   # 紫色
    "#00FFFF",   # 青色
    "#FF8000",   # 橙色
    "#8000FF",   # 靛蓝色
    "#FF0080",   # 粉红色
    "#00FF80",   # 浅绿色
    "#80FF00",   # 淡黄色
    "#0080FF",   # 天蓝色
    "#FF8080",   # 浅红色
    "#80FF80",   # 浅绿色
    "#8080FF",   # 浅蓝色
    "#FFFF80",   # 浅黄色
    "#FF80FF",   # 浅紫色
    "#80FFFF",   # 浅青色
    "#C04000",   # 深橙色
    "#4000C0"    # 深紫色
    ),
    ]
    ax1 = fig.add_subplot(122)
    labels = labels.cpu().numpy()
    for cls in range(num_labels):
        cls_idx = np.where(labels == cls)
        ax1.scatter(X_emb[cls_idx, 0], X_emb[cls_idx, 1], marker="o", color=cls_color_tuples[0][cls], label=f" CLS: {cls}", s=marker_size, alpha=1)
    ax1.legend(prop={'size': font_size})
    ax1.tick_params(axis='both', labelsize=font_size)
    ax1.set_title('Color By Class', {'fontsize' :font_size} )
    
    # title = f"PT:small{len(args.pre_train_datasets)}, Test:{args.test_dataset} embeddings" if args.dataset == 'Facebook100' else f'{args.dataset} embeddings'
    fig_name = f'small{len(args.pre_train_datasets)}_{args.test_dataset}_{kwargs["base_or_our"]}_{kwargs["trail_str"]}_{kwargs["repeat_str"]}.jpg' if args.dataset=='Facebook100' \
                else f'{args.dataset}_{kwargs["base_or_our"]}_{kwargs["trail_str"]}_{kwargs["repeat_str"]}.jpg'
    plt.savefig(f"log/emb_vis/{fig_name}", format='jpg')

# PU debias:
def get_densratio_score_matrix(dataset, device, args):
    """
    构造对比样本对 把同一个样本的两个增强看成labeled样本 其余当作Unlabeled 
    并用density ratio estimation tool来估计r_func
    用1/n当作类别先验的估计并由此来的到threshold
    也可以把threshold当作一个超参数 TBD
    最终返回所有样本对的score——matrix 
    """
    from model import drop_feature, Encoder
    from torch_geometric.utils import dropout_edge
    from main import MODEL_MAP
    features = dataset._data.x.to(device).to(dtype=torch.float32)
    edge_index = dataset._data.edge_index.to(device)
    x_1 = drop_feature(features, args.drop_feature_rate_1)
    x_2 = drop_feature(features, args.drop_feature_rate_2)
    edge_index_1 = dropout_edge(edge_index, p=args.drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=args.drop_edge_rate_2)[0]
    if args.PU_epoch > -1: # 预训练模型的表示来做Densratio Estimation
        model_path = f"/home/wangzixu/DisShiftGRACE/checkpoints/GRACE/{args.dataset}_pt_only_dim{args.PU_num_dim}/{args.PU_epoch}.pt"
        state_dict = torch.load(model_path)
        encoder = Encoder(args).to(device)
        model = MODEL_MAP[args.model](encoder, args).to(device)
        model.load_state_dict(state_dict)
        z_1 = model(x_1, edge_index_1)
        z_2 = model(x_2, edge_index_2)    
    else: # 用原始特征做r(x)估计
        features = dataset._data.x.to(device).to(dtype=torch.float32)
        edge_index = dataset._data.edge_index.to(device)
        x_1 = drop_feature(features, args.drop_feature_rate_1)
        x_2 = drop_feature(features, args.drop_feature_rate_2)
        z_1 = x_1
        z_2 = x_2
    # 拼接并查看L, U样本的分布
    n, d = z_1.shape
    
    import random
    num_sampled = n * args.num_sampled_U # int(n * np.log(n))
    U = torch.zeros(num_sampled, 2*d).to(device)
    L = torch.zeros(n, 2*d).to(device)
    U_idx_pair_list = []
    L_idx_pair_list = []
    
    for i in range(n):
        idx = i
        L_idx_pair_list.append((idx, idx))
        L[i] = torch.concat((z_1[idx], z_1[idx]))
    
    for i in range(num_sampled):
        idx_1, idx_2, = [random.randint(0, n-1) for _ in range(2)]
        U_idx_pair_list.append((idx_1, idx_2))
        U[i] = torch.concat((z_1[idx_1], z_2[idx_2]))


    print(f'# Labeled sample pair: {L.shape[0]} \n# Unlabeled sample pairs: {U.shape[0]}')
    
    from densratio import densratio
    from time import time
    start = time()
    densratio_obj = densratio(L.detach().cpu().numpy(), U.detach().cpu().numpy(), alpha=args.PU_alpha, kernel_num=args.kernel_num, 
                              lambda_range=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 
                              sigma_range=[0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
                              )  
    """
    Estimate alpha-mixture Density Ratio p(x)/(alpha*p(x) + (1 - alpha)*q(x)
    Arguments:
        x: sample from p(x).
        y: sample from q(x).
        alpha: Default 0 - corresponds to ordinary density ratio.
        sigma_range: search range of Gaussian kernel bandwidth.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        lambda_range: search range of regularization parameter for uLSIF.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        kernel_num: number of kernels. Default 100.
        verbose: indicator to print messages. Default True.

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.
    
    verbose explain: score means the squared loss of LOOCV without regularation term. the smaller the better
    """
    end = time()
    print(f'Density Ratio Estimation Time: {end-start:4f}s')
    r_func = densratio_obj.compute_density_ratio
    
    # the dict only for vis the emb of Labeld and Unlabeld(sampled) data
    LU_densratio_res_dict = {
        "r_func": r_func, 
        'U_idx_pair_list':U_idx_pair_list, 
        'L_idx_pair_list':L_idx_pair_list, 
        'z_1':z_1, 
        'z_2':z_2
    }
    
    Score_Matrix = torch.zeros((n, n)).to(device)
    start = time()
    for i in range(n):
        row_samples = torch.concat( [torch.concat([z_1[i], z_2[j]]).reshape(1, -1) for j in range(n)], dim=0)
        row_scores = r_func(row_samples.detach().cpu().numpy()) 
        Score_Matrix[i, :] = torch.tensor(row_scores).to(device)
    Score_Matrix = (Score_Matrix + Score_Matrix.t())/2 # 保持对称性
    end = time()
    print(f'Compute Score Matrix by r_x costs: {end-start:.4f}s')
    return Score_Matrix, LU_densratio_res_dict
 
def get_masks(Score_Matrix, device, args):
    n = Score_Matrix.shape[0]
    # 初始化U_Pos_masks as Score_Matrix的对角线为零
    U_Pos_mask = Score_Matrix.to(device)
    for i in range(n): # labeled samples are removed in U_Pos_mask
        U_Pos_mask[i, i] = 0
    L_mask = torch.eye(n).to(device)
    U_mask = torch.ones((n, n)).to(device) - L_mask
    
    # 得到thresholds
    U_scores = Score_Matrix[Score_Matrix > 0].reshape(-1).cpu().numpy() # Score_Matrix去除0元素并flatten成一个array
    theta_pi = None
    upper_threshold = None
    lower_threshold = None
    if args.num_threshold == 1:
        # pi = 1/n # 可能需要更好的估计class prior的方法 或者直接当作一个超参数
        num_U_pos = int(U_scores.shape[0] * args.pi) # 大于等于pi的样本量
        sorted_idx = np.argsort(U_scores)
        theta_pi = U_scores[sorted_idx][-num_U_pos] # 最大的一部分
    elif args.num_threshold == 2:
        num_lt_upper = int(U_scores.shape[0] * args.upper_percent) # 小于等于upper_percent的元素个数
        num_lt_lower = int(U_scores.shape[0] * args.lower_percent) # 小于等于lower_percent的元素个数
        sorted_idx = np.argsort(U_scores) # 升序排列
        upper_threshold = U_scores[sorted_idx][num_lt_upper] 
        lower_threshold = U_scores[sorted_idx][num_lt_lower]
    
    # 至此我们得到了未经处理过的U_Pos_mask L_mask 和 U_mask， 以及所有的thresholds
    if args.score_as_weight: # U_Pos is continuous
        if args.num_threshold == 0:
            U_Pos_max = torch.max(U_Pos_mask)
            U_Pos_min = torch.min(U_Pos_mask[U_Pos_mask > 0])
            U_Pos_mask = (U_Pos_mask - U_Pos_min) / (U_Pos_max - U_Pos_min)
            for i in range(n):
                U_Pos_mask[i, i] = 0
        elif args.num_threshold == 1:
            zero_mask = (U_Pos_mask < theta_pi)
            U_Pos_mask[zero_mask] = 0
        else: # num_threshold == 2
            deleted_samples_idx = (U_Pos_mask > lower_threshold) & (U_Pos_mask < upper_threshold)
            U_Pos_mask[deleted_samples_idx] = 0 # 保留两端，忽略中段
            
            U_Pos_max = torch.max(U_Pos_mask)
            U_Pos_min = torch.min(U_Pos_mask[U_Pos_mask > 0])
            U_Pos_mask = (U_Pos_mask - U_Pos_min) / (U_Pos_max - U_Pos_min)
            
            # deleted 和 对角线 位置上也被normalize改变了，需要重新set为0
            U_Pos_mask[deleted_samples_idx] = 0
            for i in range(U_Pos_mask.shape[0]):
                U_Pos_mask[i,i] = 0
            
            # 忽略被删除的样本，即不把它当作U
            U_mask[deleted_samples_idx] = 0
    else: # U_Pos is binary
        if args.num_threshold == 0:
            raise NotImplementedError
        elif args.num_threshold == 1:
            one_mask = (U_Pos_mask >= theta_pi)
            zero_mask = (U_Pos_mask < theta_pi)
            U_Pos_mask[one_mask] = 1
            U_Pos_mask[zero_mask] = 0
        else:
            # 中段的被忽略，U_mask中为零； lower一下的被当作负样本，U_mask中是1
            deleted_samples_idx = (U_Pos_mask > lower_threshold) & (U_Pos_mask < upper_threshold)
            U_mask[deleted_samples_idx] = 0 # 中段被忽略
            one_mask = (U_Pos_mask > upper_threshold)
            zero_mask = (U_Pos_mask <= upper_threshold)
            U_Pos_mask[one_mask] = 1
            U_Pos_mask[zero_mask] = 0 # 包括了被当作负样本的，和被忽略的
            
    # get the U_Pos_num_vec
    U_Pos_num_vec = (U_Pos_mask!=0).sum(1)
    U_Pos_num_vec[U_Pos_num_vec == 0] = -1 # 避免除零
              
    return L_mask, U_mask, U_Pos_mask, U_Pos_num_vec             
   
def vis_LU_emb(LU_densratio_res_dict, args):
    print('start vis Labeled Pos and Unlabeled data emb')
    from matplotlib import pyplot as plt
    keys = ["r_func", 'U_idx_pair_list', 'L_idx_pair_list', 'z_1', 'z_2']
    r_func, U_idx_pair_list, L_idx_pair_list, z_1, z_2 = [ LU_densratio_res_dict.get(key) for key in keys ]
    L = torch.concat( [torch.concat((z_1[pair[0]], z_2[pair[1]]) ).reshape(1, -1)  for pair in L_idx_pair_list] , dim=0)
    U = torch.concat( [torch.concat((z_1[pair[0]], z_2[pair[1]]) ).reshape(1, -1)  for pair in U_idx_pair_list] , dim=0)
    cat_data = torch.concat((U, L), dim=0)
    cat_emb = tSNE(cat_data.detach().cpu())
    num_U = U.shape[0]
    U_emb = cat_emb[:num_U]
    L_emb = cat_emb[num_U:]
    fig = plt.figure(figsize=(25, 10))
    ax_1 = fig.add_subplot(121)

    ax_1.scatter(U_emb[:, 0], U_emb[:, 1], marker='o', color="#00FFFF", label='U data (sampled)')
    ax_1.scatter(L_emb[:, 0], L_emb[:, 1], marker='o', color="#FF8000", label='L data (all)')
    ax_1.set_title('Color by Pos/Neg')

    ax_2 = fig.add_subplot(122)
    cat_scores = r_func(cat_data.detach().cpu().numpy())
    ax_2.scatter(cat_emb[:, 0], cat_emb[:, 1], c=cat_scores, cmap='Reds')
    ax_2.set_title('Color by Scores')
    plt.savefig(f'PU_emb_vis/LU_data_{args.dataset}_dim{args.PU_num_dim}_epoch{args.PU_epoch}_alpha{args.PU_alpha}.png', format='png')
    print('vis done')
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_upos', action='store_true')
    
    parser.add_argument('--llm_emb', action="store_true", help="Use llm to enhance the TAG feature")
    parser.add_argument('--llm_name', type=str, choices=['Llama-3.2-1B', 'Llama-3.2-3B', 'Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Qwen2.5-3B', 'Qwen2.5-7B'], default='Llama3.2-1B')
    parser.add_argument('--llm_emb_dim', type=int, default=1024)
    
    parser.add_argument('--beta', type=float, default=0, help='weight of DU+ contrastive loss; in case of too large weight for DU+')
    
    parser.add_argument('--use_gpt', action="store_true", help='Use TAPE embedding with GPT-responses')
    
    parser.add_argument('--bert_feat', action='store_true', help='Use Bert embedding on Original Title-Abstract text')
    parser.add_argument('--bert_version', type=str, choices=['uncased', 'cased'], default='cased')
    
    parser.add_argument('--tape_feat', action="store_true", help='Use TAPE embedding as Bert-Enhansed features')
   
    parser.add_argument('--giant_feat', action="store_true", help='Use GIANT-XRT embedding as Bert-Enhansed features')
    
    parser.add_argument('--n_trials', type=int, default=20, help='the max number of trials for hyper-params-tuning')
    parser.add_argument('--cur_repeat', type=int, help='Only Used when dataset is WikiCS')
    
    # senario: OOD or Normal SSL
    parser.add_argument('--senario', type=str, choices=['OOD', 'ID'], default='ID')
    
    
    # update the infonce from theroy view
    parser.add_argument('--theroy_view', action="store_true", help='log(L+U_Pos) to log(L)+log(U_Pos)')
    parser.add_argument('--start_debias_epoch', type=int, default=200)
    parser.add_argument('--update_interval', type=int, default=1)
    # 逐行选取U+ 或者 全局选取U+ -> 对应逐行norm 或者 全局 norm
    parser.add_argument('--norm_sim_matrix', type=str, choices=['global', 'line_by_line'], default='global')
    
    # sim martix 得到权重矩阵W 对样本相似度 或者lij加权
    parser.add_argument('--B_W_threshold', type=float, default=0.6)
    parser.add_argument('--R_W_threshold', type=float, default=0.6)
    
    # Between_View的sim_matrix norm后是否保持对角线全1 -> 对应着是否要对原GRACE的正样本筛选， 1: 不筛选，保持原有正样本； 0: 和U一样对原有正样本进行筛选
    # 默认要保持对角线全一，即对原正样本不筛选
    parser.add_argument('--stay_diag_eye', type=int, choices=[0, 1], default=1, help='1: B_W在norm后保持对角线全1   0: 不用保持')
    
    # only for analysis the homo-ratio of the learned representation
    parser.add_argument('--best_W_threshold',type=float)  
    
    """
    for read_log under repeat PT & SFT setting
    """
    # read top-k trials with highest val_acc
    parser.add_argument('--top_k_val_trials', type=int, default=20)
    # checkpoint_dir & log_file_path 
    parser.add_argument('--checkpoint_dir', type=str, default='/home/wangzixu/DisShiftGRACE/checkpoints/GRACE/Cora-Weighted_Sum_LogP_Theroy_view')
    parser.add_argument('--log_file_path', type=str, default='/home/wangzixu/DisShiftGRACE/Theroy_View_Log/Cora_GRACE_Theroy_View_R_U+_Large_Thresholds.log')
    
    # Use PU to debias 
    # pre_train only to get a low-dim Representation for Density Ratio Estimation
    parser.add_argument('--pt_only', action="store_true", help="Only pre_train, just for get a good representation for density ratio estimation")
    parser.add_argument('--PU', action='store_true', help='Use PU to debias if set true')
    parser.add_argument('--PU_alpha', default=0, type=float, help="RuLSIF 中的alpha 不同的alpha其实得到的密度估计得到的r_alpha(x)也可以反应p(y=1|x)的序")
    parser.add_argument('--PU_num_dim', type=int, default=64 ,choices=[128, 64, 32], help='PU debias 用来估计r(x)的表示的维度')
    parser.add_argument('--PU_epoch', type=int, default=499, choices=[-1, 0, 100, 200, 300, 400, 499], help='PU debias 用来估计r(x)的模型epoch数-1; -1: 原始特征； i: i+1个训练后的模型得到的表示')
    parser.add_argument('--kernel_num', type=int, default=100, help="RuLSIF 's kernel numbers ")
    parser.add_argument('--num_sampled_U', default=3, type=int, help='在估计r(X)时U数据采样是L的多少倍')
    # r(x) score as Pu+ weight also affect the U_num_vec
    # U_Pos_mask binary / continuous
    parser.add_argument('--score_as_weight', action="store_true", help="if true the U_Pos_mask is the normalization of r(x) matrix else use  pi to get theta_pi as a threshold and U_Pos_mask is a 0-1 binary mask; U_Pos_mask的值是continuous (True)的还是binary (False)的")
    # Class-prior: 在修正的P+ P-中使用，此外在score_as_weight为false时用于估计threshold
    parser.add_argument('--pi', type=float, default=0.005, help="决定了U中有大概有多少比例的样本被分类为正样本, 类似Nap中的相似度的thereshold")
    # 表示的相似度代替 Score-Matrix 随着模型不断更新并得到Masks
    parser.add_argument('--sim_as_score',action="store_true")
    parser.add_argument('--start_PU_epoch', type=int, default=0, help='默认为0 只有在sim_as_score时需要设定')
    # 指示量，用来提示encoder是在做Density Ratio Estimation 还是在后续的SSL
    parser.add_argument('--estimate', type=int, default=0, help="if 1 the model's encoder's out_dim is PU_num_dim; if 0 then is num_hidden")
    # vis PU embeding
    parser.add_argument('--vis_PU', action='store_true', help='vis PU data emb or not')
    # U+ loss mean or sum
    parser.add_argument('--u_pos_loss', type=str, default='sum', choices=['sum', 'mean'])
    # num of threshold
    parser.add_argument('--num_threshold', type=int, default=0, choices=[0, 1, 2], help='0: 没有threshold，只适用于连续的U_Pos_mask; 1: 0-1的mask，就是theta——pi; 2: upper和lower，连续、binary的U_Pos_mask都可以用')
    parser.add_argument('--upper_percent', type=float, default=0.7, help='U_Pos中大于百分之upper_percent的被保留')
    parser.add_argument('--lower_percent', type=float, default=0.3, help='U_Pos中小于lower_percent的被保留, 类似于用pi来计算theta_pi')
    
    # which model to use
    parser.add_argument('--model', type=str, default='GRACE', choices=['GRACE', 'DGI', 'BGRL', 'COSTA', 'MVGRL', 'GCA', 'VGAE', 'GCN', 'GAT', 'GBT'])
    
    # test mode is for load checkpoint only
    parser.add_argument('--test_mode', action='store_true', help='Test mode in Load checkpoint for ood-test; else Pre_train+SFT+SFT_test+Valid')
    
    # only for FB dataset; Using which datasets for pre-training(also SFT and SFT_test), validation and test;
    parser.add_argument('--pre_train_datasets', type=str, nargs="+", default=
    [
    'Amherst41',  
    'Bowdoin47',  
    'Hamilton46',  
    'Trinity100',  
    'USFCA72',  
    'Williams40',  
    'Oberlin44',  
    'Smith60',  
    'Wellesley22',  
    # 'Vassar85',  
    # 'Middlebury45',  
    # 'Pepperdine86',  
    # 'Colgate88',  
    # 'Santa74',  
    # 'Wesleyan43',  
    # 'Mich67',  
    # 'Bucknell39',  
    # 'Brandeis99',  
    # 'Howard90',  
    # 'Rice31',  
    # 'Rochester38',  
    # 'Lehigh96',  
    # 'Johns Hopkins55',  
    # 'Wake73',  
    # 'American75',  
    # 'MIT8',  
    # 'William77',  
    # 'UChicago30',  
    # 'Princeton12',  
    # 'Carnegie49',  
    # 'Tufts18',  
    # 'UC64',  
    # 'Vermont70',  
    # 'Emory27',  
    # 'Dartmouth6',  
    # 'Tulane29',  
    # 'WashU32',  
    # 'Villanova62',  
    # 'Vanderbilt48',  
    # 'Yale4',  
    # 'Brown11',  
    # 'UCSC68',  
    # 'Maine59',  
    # 'Georgetown15',  
    # 'Duke14',  
    ])
    parser.add_argument('--valid_dataset',
                        type=str,
                        default=  
                        "Emory27",
                        help="the valid dataset, just like the OOD-validation in GOOD; for tuning the hyper_params")
    parser.add_argument('--test_dataset',
                        type=str,
                        default=  
                        "Vassar85",
                        help="the test dataset, just like the OOD-test in GOOD; for test the models' ood-generalization ability")
    
    # choice of dataset
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name for OOD: [Facebook100_5-8K-nodes, GOODTwitch, GOODCBAS, GOODHIV]; else: [Cora, CiteSeer, PubMed]')
    parser.add_argument('--domain_flag', type=str, default='degree', help='GOODCora 的domain划分依据', choices=['word', 'degree'])
    # where to load the FB data
    parser.add_argument('--datasets_dict_path', type=str, default='/home/wangzixu/data/datasets_dict_2000-10000nodes',
                        help='the path of datasets_dict which contains the pre-processed datasets_dict;'
                             'on 10.61.1.25: /home/wangzixu/data/datasets_dict_2000-10000nodes'
                             'on 10.208.62.15: /data/wangzixu/datasets_dict_2000-10000nodes')


    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--save_model', action="store_true",help="Save the model or not.")
    parser.add_argument('--repeat_times', type=int, default=3)
    parser.add_argument('--current_repeat_times', type=int)

    parser.add_argument('--task', type=str, default="node-cls", help="the task to do",
                        choices=["node-cls", "link-pred", "graph-cls"])
    
    # hyper-params of weight of Three losses; Deprecated Now
    # parser.add_argument('--contrastive_loss_weight', type=float, default=1)
    # parser.add_argument('--domain_loss_type', type=str, default="MMD", choices=["Center", "MMD"])
    # parser.add_argument('--domain_loss_weight', type=float, default=0)
    # parser.add_argument('--gen_loss_weight', type=float, default=0)
    # parser.add_argument('--gen_loss_type', type=str, default="GAN", choices=["GAN", "GRL"]) # GRL: Gradient Reversal Layer

    
    # hyper-params of Model
    parser.add_argument('--seed', type=int, default=39788) # 1001， 
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--activation', type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument('--base_model', type=str, default="GCNConv", choices=["GCNConv", "SAGEConv"])
    parser.add_argument('--num_layers', type=int, default=2)
    
    # hyper-params of GCL Pre-training
    
    # shared
    parser.add_argument('--num_epochs', type=int, default=500) # fixed
    
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--tolerance_epoch_num', type=int, default=10)
    parser.add_argument('--gradient_drop_threshold', type=float, default=1e-3, choices=[1e-3, 1e-4, 1e-5]) # 1e-3 for GRACE, COSTA, GCA, GBT; 1e-4 for DGI, BGRL; 1e-5 for MVGRL;
    
    # GRACE, COSTA
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
    
    # COSTA
    parser.add_argument('--costa_k_ratio', type=float, default=0.5)
    
    # BGRL
    parser.add_argument('--moving_average_decay', type=float, default=0.99)
    
    # MVGRL
    parser.add_argument('--mvgrl_alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=0, help='0: 不分batch  >0: batch的大小')   
    
    # GCA
    parser.add_argument('--drop_scheme', type=str, default="degree", choices=['uniform', 'degree', 'evc', 'pr'])
    
    # GBT
    parser.add_argument('--p_x', type=float, default=0.1)
    parser.add_argument('--p_e', type=float, default=0.4)
    
    # hyper-params of Our Method
    parser.add_argument('--add_pos_pair',  action="store_true", help="Run or not.")# whether to add the positive pair after selecting the removed negative pair
    parser.add_argument('--rv_neg_ratio', type=float, default=0) # the ratio of cross domain negative samples removed
    parser.add_argument('--start_rv_epoch', type=int, default=0) # when to start remove the cross-domain samples
    
    # Visulization of Masks and Embeddings
    parser.add_argument('--vis_mask', action="store_true", help="whether to vis the mask of the removed negative samples")
    parser.add_argument('--vis_emb', action="store_true", help="whether to vis the embeddings of the pre-train-domains and test domains")
    
    # Whether tune the hyper params using Optuna
    parser.add_argument('--tune_hyper_params', action='store_true', help="whether to tune the hyper param")
    
    # hyper-params to load the checkpoint; 
    """
    1. --model_path only
    2. trail, pt_num, test_domain, add_pos, ratio, start
    """
    parser.add_argument('--model_dir', type=str, help='Test all the models under this dir if model_path is not given', default='checkpoints/GOODTwitch_addPossFalse')
    
    parser.add_argument('--model_path', type=str, help="Model path to choose checkpoint", default=None)
    
    parser.add_argument('--trail', type=int, help='Trail number of the checkpoint to load')
    parser.add_argument('--load_pt_num', type=int, help='FB100 pre_train_datasets num, to choose the loaded checkpoint')
    parser.add_argument('--load_test', type=str, help='Test domain to choose the checkpoint')
    parser.add_argument('--load_add_pos', action="store_true", help='Add positive samples or not, to choose the checkpoint')
    parser.add_argument('--load_ratio', type=float, help="Remove neg samples ratio, to choose the checkpoint")
    parser.add_argument('--load_start', type=int, help="Start remoing epoch, to choose the checkpoint")
    
    
    args = parser.parse_args()
    return args




if __name__ == "__main__":
   pass