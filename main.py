import sys
import os
from pre_train import *
from model import *
from eval import *
from utils import *
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import pickle
import pdb
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
from time import time
from torch_geometric.data import Dataset

#  工具类
class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self._data = data
    def len(self):
        return 1
    def get(self, idx):
        return self._data


INT_MAX = 2 ** 31 - 1

MODEL_MAP = {
 'GRACE': GRACE,
 'DGI': DGI,  
 'BGRL': BGRL,
 'COSTA': COSTA,
 'MVGRL': MVGRL,
 'GCA': GCA,
 'VGAE': VGAE,
 'GAE': GAE,
 'GBT': GBT
}

def init_encoder(args):
    if args.model == 'MVGRL':
        return MVGRL_Encoder(args)
    elif args.model == 'VGAE':
        return VariationalGCNEncoder(args)    
    else:
        return Encoder(args)
    
def init_model(encoder: nn.Module, args):
    if args.model == 'VGAE':
        return VGAE(encoder)
    else:
        return MODEL_MAP[args.model](encoder, args)

# 完全相同的数据，只是随机种子不同，从头预训练、微调、valid(Test) n 次；
def repeat_n_times_from_scrach(_dataset, args, device, trial=None):
    # set random seed
    set_random_seed(args.seed)
    
    '''
        NaP超参数的要求
        1. add_pos_pair为True则rv_neg_ratio和start_rv_epoch必须大于零
        2. add_pos_pair为False则两者必须等于零
    '''
    # if args.theroy_view:
    #     assert ((args.start_debias_epoch >= 0) and (args.start_rv_epoch >= 0) ) ,"超参数Ratio、Start_Epoch设置异常"
    #     #     or ((args.add_pos_pair is not True) and (args.rv_neg_ratio == 0) and (args.start_rv_epoch == 0)),\
        
    # print(f'shared params: \nlearning_rate: {args.learning_rate}, weight_decay: {args.weight_decay}')
    # if args.model == 'GRACE':
    #     print(f"hyper params: tau: {args.tau}, dropEdgeRate: ({args.drop_edge_rate_1}, {args.drop_edge_rate_2}), dropFeatRate: ({args.drop_feature_rate_1},{args.drop_feature_rate_2}) \
    #            \nstart_debias_epoch: {args.start_debias_epoch} update_interval: {args.update_interval} W_threshold:{args.W_threshold}"
    #            )
    # elif args.model == 'BGRL':
    #     print(f"hyper params:\ndropEdgeRate: ({args.drop_edge_rate_1}, {args.drop_edge_rate_2}), dropFeatRate: ({args.drop_feature_rate_1},{args.drop_feature_rate_2})")
    # elif args.model == 'COSTA':
    #     print(f"hyper params:\nk_ratio: {args.costa_k_ratio}, tau: {args.tau},dropEdgeRate: ({args.drop_edge_rate_1}, {args.drop_edge_rate_2}), dropFeatRate: ({args.drop_feature_rate_1},{args.drop_feature_rate_2})")
    # elif args.model == 'MVGRL':
    #     print(f"hyper params:\nmvgrl_alpha: {args.mvgrl_alpha}")
    
    print(f'*'*200)
    print('PipeLine Begins.')
    if trial is not None:
        args.trial_number = trial.number # trial 信息放在args里
        print(f'Trial Number: {trial.number}')

    # 估计r(x) 以及 得到U_Pos_masks, U_Pos_num_vec相关的参数，方便后续save model; 如果PU为false则为空字符串
    r_params = '' if not args.PU else f'{args.dataset}_dim{args.PU_num_dim}_epoch{args.PU_epoch}_alpha{args.PU_alpha}_kernelNum{args.kernel_num}_numSamU{args.num_sampled_U}'
        
    # PU pre-process:  RuLSIF estimate the density ratio: Pl / Pu 即r(x), 并得到L，U，U_Pos masks 以及 U_Pos_num_vec 放在这里是为了在调参数的时候也能一起调
    # We don't use PU to update the InfoNCE anymore
    # if args.PU and not args.sim_as_score:
    #     import os
    #     Score_Matrix_path = 'Score_Matrix'  
    #     if not os.path.exists(Score_Matrix_path):
    #         os.makedirs(Score_Matrix_path)
        
    #     Score_Matrix_file = f'{Score_Matrix_path}/{r_params}.pt'
    #     if os.path.exists(Score_Matrix_file):
    #         Score_Matrix = torch.load(Score_Matrix_file).to(device)
        
    #         y = _dataset[0].y
    #         Label_Matrix = (y.unsqueeze(1) == y).int().to(device)
            
    #     else:     
    #         args.estimate = 1
    #         Score_Matrix, LU_densratio_res_dict = get_densratio_score_matrix(_dataset, device, args)
    #         args.estimate = 0
    #         torch.save(Score_Matrix, Score_Matrix_file)
        
    #         # vis L and U (sampled ) data 's feature color by Pos/Neg and DensityRatioScore
    #         if args.vis_PU:
    #             vis_LU_emb(LU_densratio_res_dict, args)
        

    #     # get the masks based on the Score_Matrix
    #     L_mask, U_mask, U_Pos_mask, U_Pos_num_vec = get_masks(Score_Matrix, device, args)
    
    """
    for OOD senario: 
        Pre-train on PT,
        SFT on SFT
        SFT-test on SFT_Test,
        Test on Test
    
    for ID senario:
        Pre-train on the whole dataset; 
        SFT on train, Val on val, Test on test
    
    """
    
    # Pre-train, sft and test several times
    # emb_pdd_list = []
    # emb_inner_dis_list = []
    sftTest_res_list = []
    valid_res_list = []

    # Task header and pre-train model which has the best test acc
    best_SFT_clf = None
    best_PT_model_state = None
    for repeat_time in range(args.repeat_times):
        args.repeat_time = repeat_time # repeat time 信息放在args里
        encoder = init_encoder(args).to(device)
        model = init_model(encoder, args).to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        trainer = Trainer(dataset=_dataset, pt_model=model, optimizer=optimizer, device=device,args=args,)
        #   Old Params for PU-debias framework
        #   L_mask=L_mask if args.PU and not args.sim_as_score else None, 
        #   U_mask=U_mask if args.PU and not args.sim_as_score else None, 
        #   U_Pos_mask=U_Pos_mask if args.PU and not args.sim_as_score else None, 
        #   U_Pos_num_vec=U_Pos_num_vec if args.PU and not args.sim_as_score else None,
        #   z_1 = LU_densratio_res_dict['z_1'] if args.PU else None,
        #   z_2 = LU_densratio_res_dict['z_2'] if args.PU else None,
                          
        gradient_up_num = 0
        last_epoch_loss = INT_MAX
        best_sftTest_acc = 0
        best_valid_acc = 0
        import sys
        # Pre-train
        for epoch in (pbar := tqdm(range(0, args.num_epochs), ncols=100)):
            pbar.set_description(f"Epoch: {epoch}")
            loss = trainer.train()
            pbar.set_postfix(loss=loss.item(), gradient_not_down=gradient_up_num)
            
            # Early Stop
            if last_epoch_loss - loss < args.gradient_drop_threshold: # loss decrease 需要超过阈值才能算一个epoch训练有效
                gradient_up_num += 1
                if gradient_up_num == args.tolerance_epoch_num:
                    tqdm.write(f"loss has not decreased for {args.tolerance_epoch_num} epochs, so we stop training")
                    break
            else:
                gradient_up_num = 0
            last_epoch_loss = loss

            # if args.pt_only: # for PU-debiased framework
            #     # 保存预处理步骤的模型，只需要预训练得到表示的话，那就保存100的整数倍的epoch;
            #     # 用这个表示 和 RuLSIF来估计pl/pu 
            #     if epoch % 100  == 0 or epoch == args.num_epochs-1:
            #         import os
            #         model_path = f'checkpoints/{args.model}/{args.dataset}_pt_only_dim{args.num_hidden}'
            #         if not os.path.exists(model_path):
            #             os.makedirs(model_path)
            #         torch.save(model.state_dict(), f'{model_path}/{epoch}.pt')
            #     continue
            
            # SFT,SFT_test & validation; Save the PT_model and cls_head; 
            """不同的senario有不同的设定
            Validation for tune the hyper-params
            SFT_test for record the SFT test acc not for tune hyper params
            ID: PT on all; SFT on train_mask; SFT_test on val_mask; Validation on val_mask; Test on test_mask;
            OOD: PT on PT_mask; SFT on SFT_mask; SFT_test on SFT_test_mask; Validation on Valid_mask; Test on Test_mask;
            """
            if epoch % 10 == 0 or epoch == args.num_epochs - 1:
                # 只需要在最后一轮计算pdd和inner_dis
                # if epoch == args.num_epochs - 1 and args.senario == 'OOD':
                #     # pdd
                #     pdd = trainer.pdd()
                #     # tqdm.write(f'epoch {epoch} pdd: {pdd:.6f}')
                #     emb_pdd_list.append(pdd.item())
                    
                #     # inner_distance:
                #     inner_dis = trainer.mean_inner_distence()
                #     emb_inner_dis_list.append(inner_dis)
                
                # SFT and SFT-Test; the SFT-Test is just for record, not used for tune_hyper_params
                # OOD: SFT_test != Valid_mask & SFT_test is ID with PT; 
                # ID: SFT_test == Valid_mask
                sftTest_res = fine_tune(model, dataset=_dataset, device=device, args=args)
                
                # Update the val_acc
                if sftTest_res['acc'] > best_sftTest_acc: 
                    best_sftTest_acc = sftTest_res['acc']
                    tqdm.write(f'Epoch{epoch} best SFT_test acc of repeat{repeat_time}: { best_sftTest_acc :.4f}')
                
                # Use the model and the sft-head
                # valid_res = Valid_or_Test(model, sftTest_res['clf'], dataset=_dataset, device=device, args=args)
                # # Update, Record  the test_acc & Save the model
                # if valid_res['acc'] > best_valid_acc:
                #     best_valid_acc = valid_res['acc']
                #     tqdm.write(f'Epoch{epoch} best test acc of repeat{repeat_time}: {best_valid_acc:.4f}')
                #     # 保存当前最优模型
                #     if args.save_model:
                #         # PU-debias framework需要的
                #         # threshold_params = f'numThreshold{args.num_threshold}'
                #         # if args.num_threshold == 1:
                #         #     threshold_params += f'_pi{args.pi}'
                #         # elif args.num_threshold == 2:
                #         #     threshold_params += f'_upperPercent{args.upper_percent}_lowerPercet{args.lower_percent}'
                #         # else:
                #         #     pass
                #         # mask_Binary_Continous_param = f'{"continuous" if args.score_as_weight else "binary"}-UPosMask'    
                        
                #         # 在test上表现最好的预训练模型和分类头
                #         best_PT_model_state = model.state_dict()
                #         best_SFT_clf = sftTest_res['clf']
                #         import os
                #         if args.senario == 'OOD':
                #             # if args.dataset == 'Facebook100':
                #             #     model_path = f'checkpoints/{args.model}/small{len(args.pre_train_datasets)}_valid{args.valid_dataset}_addPoss{args.add_pos_pair}'
                #             # else:
                #             #     model_path = f'checkpoints/{args.model}/{args.dataset}_addPoss{args.add_pos_pair}'
                #             if args.theroy_view:
                #                 theroy_view_params = f'_start_debias_epoch{args.start_debias_epoch}_update_interval{args.update_interval}_R_W_threshold{args.R_W_threshold}_B_W_threshold{args.B_W_threshold}'
                #                 model_path = f'checkpoints/{args.model}/{args.dataset}_{args.senario}-Weighted_Sum_LogP_Theroy_view'
                #             else:
                #                 model_path = f'checkpoints/{args.model}/{args.dataset}_{args.senario}-Baseline'
                                
                #         else: # ID  senario
                #             if args.theroy_view:
                #                 theroy_view_params = f'_start_debias_epoch{args.start_debias_epoch}_update_interval{args.update_interval}_R_W_threshold{args.R_W_threshold}_B_W_threshold{args.B_W_threshold}'
                #                 model_path = f'checkpoints/{args.model}/{args.dataset}-Weighted_Sum_LogP_Theroy_view'
                #             else:
                #                 model_path = f'checkpoints/{args.model}/{args.dataset}-Baseline'
                #                 # if args.sim_as_score:
                #                 #     model_path = f'checkpoints/{args.model}/{args.dataset}' + (f'-PU_debiased/Sim_As_Score/{mask_Binary_Continous_param}_{threshold_params}_uPosLoss-{args.u_pos_loss}' if args.PU else '')
                #                 # else:
                #                 #     model_path = f'checkpoints/{args.model}/{args.dataset}' + (f'-PU_debiased/Density_Ratio_Estimation/{r_params}_{mask_Binary_Continous_param}_{threshold_params}_uPosLoss-{args.u_pos_loss}' if args.PU else '')

                #         if not os.path.exists(model_path):
                #             os.makedirs(model_path)
                            
                #         pt_model_name = f'PT_trial{trial.number if trial is not None else "None"}_repeat{repeat_time}_tau{args.tau}_dropEdgeRate{args.drop_edge_rate_1}-{args.drop_edge_rate_2}_dropFeatRate{args.drop_feature_rate_1}-{args.drop_feature_rate_2}{theroy_view_params if args.theroy_view else ""}.pt'
                #         sft_head_name = pt_model_name.replace('PT','SFT')
                #         torch.save(best_PT_model_state, f'{model_path}/{pt_model_name}')
                #         with open(f'{model_path}/{sft_head_name}', 'wb') as f:
                #             pickle.dump(best_SFT_clf, f)                          
                                
        # Vis the Pre-Trained Embeddings of pre-train-domains and test domains
        if args.vis_emb and repeat_time == args.repeat_times - 1:
            best_encoder = Encoder(args).to(device)
            best_model = MODEL_MAP[args.model](best_encoder, args).to(device)
            best_model.load_state_dict(best_PT_model_state)
            vis_pt_test_embs(dataset=_dataset, pt_model=best_model, args=args, device=device, aug_1_dict=None, aug_2_dict=None,  epoch=epoch)

        sftTest_res_list.append({'acc': best_sftTest_acc})
        valid_res_list.append({'acc': best_valid_acc})

    #  Compute the mean and std of PDD, SFT and Test
    repeat_statistic(sftTest_res_list, valid_res_list, args=args)
    valid_acc_mean = np.mean([valid_res['acc'] for valid_res in valid_res_list])
    return valid_acc_mean

# 预训练一次，微调时valid、test的划分只有比例保持一致，具体方式不同，重复n次
def repeat_n_times_sft(_dataset, args, device, trial=None):
    # set random seed
    set_random_seed(args.seed)
    print(f'*'*200)
    print('PipeLine Begins.')
    if trial is not None:
        args.trial_number = trial.number # trial 信息放在args里
        print(f'Trial Number: {trial.number}')
    
    """
    for OOD senario: 

    for ID senario:
        Pre-train on the whole dataset; 
        SFT on train, Val on val, Test on test
    """
    
    # Task header and pre-train model which has the best test acc
    best_SFT_clf_list = [None, None, None]
    best_PT_model_state = None
    
    encoder = init_encoder(args).to(device)
    model = init_model(encoder, args).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Trainer(dataset=_dataset, pt_model=model, optimizer=optimizer, device=device,args=args,)
        
    gradient_up_num = 0
    last_epoch_loss = INT_MAX
    
    # 设置不同场景下的统计量
    if args.senario == 'ID':
        # 默认IID场景下需要统计的：（IID）val-acc & test-acc
        best_val_acc = 0
        val_acc_std = 0
        
        corresponding_test_acc = 0
        test_acc_std = 0
    # OOD场景下的 OOD-val-acc & OOD-test-acc
    elif args.senario == 'OOD':
        best_OOD_val_acc = 0
        OOD_val_acc_std = 0
        
        corresponding_OOD_test_acc = 0
        OOD_test_acc_std = 0
        corresponding_val_acc = 0
        val_acc_std = 0
        corresponding_test_acc = 0
        test_acc_std = 0
    else:
        raise NotImplementedError('No such senario')
    
    import sys
    # Pre-train 1 次，每10 epoch 检测一下
    for epoch in (pbar := tqdm(range(0, args.num_epochs), ncols=100)):
        pbar.set_description(f"Epoch: {epoch}")
        loss = trainer.train()
        pbar.set_postfix(loss=loss.item(), gradient_not_down=gradient_up_num)
        
        # Early Stop
        if last_epoch_loss - loss < args.gradient_drop_threshold: # loss decrease 需要超过阈值才能算一个epoch训练有效
            gradient_up_num += 1
            if gradient_up_num == args.tolerance_epoch_num:
                tqdm.write(f"loss has not decreased for {args.tolerance_epoch_num} epochs, so we stop training")
                break
        else:
            gradient_up_num = 0
        last_epoch_loss = loss
        
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            repeat_val_acc = []
            repeat_test_acc = []
            sft_head_list = []
            if args.senario == 'OOD':
                repeat_OOD_val_acc = []
                repeat_OOD_test_acc = []
            # 固定PT，多次SFT on 不同的划分
            for repeat in range(args.repeat_times):
                args.cur_repeat = repeat if args.dataset == 'WikiCS' else None
                sft_res = fine_tune(model, dataset=_dataset, device=device, args=args)
                repeat_val_acc.append(sft_res['val_acc'])
                repeat_test_acc.append(sft_res['test_acc'])
                sft_head_list.append(sft_res['clf'])
                
                if args.senario == 'OOD': # 如果是OOD场景，则相当于多测组OOD的val和OOD的test
                    #OOD val and test's res
                    OOD_val_res = Valid_or_Test(model, sft_res['clf'], dataset=_dataset, device=device, args=args)
                    repeat_OOD_val_acc.append(OOD_val_res['acc'])
                    
                    args.test_mode = True # 测试
                    OOD_test_res = Valid_or_Test(model, sft_res['clf'], dataset=_dataset, device=device, args=args)
                    args.test_mode = False # 恢复到非测试状态，不影响之后的OOD-val
                    repeat_OOD_test_acc.append(OOD_test_res['acc'])
                
            mean_val_acc = np.mean(repeat_val_acc)
            mean_test_acc = np.mean(repeat_test_acc)
            
            # 根据Val来存储模型，OOD看OOD_val的结果； ID看val的结果
            if args.senario == 'OOD':
                mean_OOD_val_acc = np.mean(repeat_OOD_val_acc)
                mean_OOD_test_acc = np.mean(repeat_OOD_test_acc)
                if mean_OOD_val_acc > best_OOD_val_acc: # 根据OOD_val的结果来保存模型
                    best_OOD_val_acc = mean_OOD_val_acc
                    OOD_val_acc_std = np.std(repeat_OOD_val_acc)
                    tqdm.write(f'Epoch{epoch} best OOD_val acc: { best_OOD_val_acc :.4f}')
                    # 记录val acc最大时对应的test acc； 而不是全局最大的test-acc
                    corresponding_OOD_test_acc = mean_OOD_test_acc
                    OOD_test_acc_std = np.std(repeat_OOD_test_acc)
                    tqdm.write(f'Epoch{epoch} corresponding OOD test acc: { corresponding_OOD_test_acc :.4f}')
                    
                    # 更新一下最好OOD-val时对应的ID val 和test acc
                    corresponding_val_acc = mean_val_acc
                    val_acc_std = np.std(repeat_val_acc)
                    corresponding_test_acc = mean_test_acc
                    test_acc_std = np.std(repeat_test_acc)
                    
                    if args.save_model:
                        best_PT_model_state = model.state_dict()
                        best_SFT_clf_list = sft_head_list
                        import os
                        model_path = f'/home/wangzixu/DisShiftGRACE/checkpoints_repeat-SFT-Only/{args.model}/{args.dataset}_{args.senario}{"-Theroy_View" if args.theroy_view else "-Baseline"}'
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        
                        theroy_view_params = f'_start{args.start_debias_epoch}_interval{args.update_interval}_R_threshold{args.R_W_threshold}_B_threshold{args.B_W_threshold}'
                        
                        pt_model_name = f'PT_trial{trial.number if trial is not None else "None"}_tau{args.tau}_dropEdgeRate{args.drop_edge_rate_1}-{args.drop_edge_rate_2}_dropFeatRate{args.drop_feature_rate_1}-{args.drop_feature_rate_2}{theroy_view_params if args.theroy_view else ""}.pt'
                        sft_head_name = pt_model_name.replace('PT','SFT')
                        torch.save(best_PT_model_state, f'{model_path}/{pt_model_name}')
                        for i,sft_head in enumerate(best_SFT_clf_list):
                            with open(f'{model_path}/{sft_head_name}_repeat{i}', 'wb') as f:
                                pickle.dump(sft_head, f)    
            else:  # ID场景下，根据val-acc 来保存模型 
                # Update the best val / test acc and the correspond std
                if mean_val_acc > best_val_acc: 
                    best_val_acc = mean_val_acc
                    val_acc_std = np.std(repeat_val_acc)
                    tqdm.write(f'Epoch{epoch} best val acc: { best_val_acc :.4f}')
                    # 记录val acc最大时对应的test acc； 而不是全局最大的test-acc
                    corresponding_test_acc = mean_test_acc
                    test_acc_std = np.std(repeat_test_acc)
                    tqdm.write(f'Epoch{epoch} corresponding test acc: { corresponding_test_acc :.4f}')
                    
                    if args.save_model:
                        best_PT_model_state = model.state_dict()
                        best_SFT_clf_list = sft_head_list
                        import os
                        model_path = f'/home/wangzixu/DisShiftGRACE/checkpoints_repeat-SFT-Only/{args.model}/{args.dataset}{"-Theroy_View" if args.theroy_view else "-Baseline"}_More_Val'
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        
                        theroy_view_params = f'_start{args.start_debias_epoch}_interval{args.update_interval}_R'
                        
                        pt_model_name = f'PT_trial{trial.number if trial is not None else "None"}_tau{args.tau}_dropEdgeRate{args.drop_edge_rate_1}-{args.drop_edge_rate_2}_dropFeatRate{args.drop_feature_rate_1}-{args.drop_feature_rate_2}{theroy_view_params if args.theroy_view else ""}.pt'
                        sft_head_name = pt_model_name.replace('PT','SFT')
                        torch.save(best_PT_model_state, f'{model_path}/{pt_model_name}')
                        for i,sft_head in enumerate(best_SFT_clf_list):
                            with open(f'{model_path}/{sft_head_name}_repeat{i}', 'wb') as f:
                                pickle.dump(sft_head, f)    
                            

    print(f"Seed: {args.seed}")
    if args.dataset == 'Facebook100':
        print(f"PT,SFT,SFT_test: {args.pre_train_datasets}")
        print(f"Valid: {args.valid_dataset}")
    else:
        print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Num_epochs: {args.num_epochs}")
    print(f"Repeat_times: {args.repeat_times}")
    if args.tune_hyper_params:
        print(f'Trial_number: {args.trial_number}')
    if args.model == 'GRACE':
        print(f'tau: {args.tau}')
        print(f'drop_edge_rate_1: {args.drop_edge_rate_1}')
        print(f'drop_edge_rate_2: {args.drop_edge_rate_2}')
        print(f'drop_feature_rate_1: {args.drop_feature_rate_1}')
        print(f'drop_feature_rate_2: {args.drop_feature_rate_2}')
    elif args.model == 'GCA':
        print(f'tau: {args.tau}')
        print(f'drop_edge_rate_1: {args.drop_edge_rate_1}')
        print(f'drop_edge_rate_2: {args.drop_edge_rate_2}')
        print(f'drop_feature_rate_1: {args.drop_feature_rate_1}')
        print(f'drop_feature_rate_2: {args.drop_feature_rate_2}')
        print(f'drop_scheme: {args.drop_scheme}')
    if args.theroy_view:
        print(f'start_debias_epoch: {args.start_debias_epoch}')
        print(f'update_interval: {args.update_interval}')
        print(f'R_W_threshold: {args.R_W_threshold}')
        print(f'B_W_threshold: {args.B_W_threshold}')
    if args.senario =='ID':
        print(f'best val acc: {best_val_acc*100:.4f}% ± {val_acc_std*100:2f}%')
        print(f'correspoding test acc: {corresponding_test_acc*100:.4f}% ± {test_acc_std*100:2f}%')
        tune_object = best_val_acc
    elif args.senario == 'OOD':
        print(f'best OOD-val acc: {best_OOD_val_acc*100:.4f}% ± {OOD_val_acc_std*100:2f}%')
        print(f'corresponding OOD-test acc: {corresponding_OOD_test_acc*100:.4f}% ± {OOD_test_acc_std*100:2f}%')
        print(f'corresponding val acc: {corresponding_val_acc*100:.4f}% ± {val_acc_std*100:2f}%')
        print(f'corresponding test acc: {corresponding_test_acc*100:.4f}% ± {test_acc_std*100:2f}%')
        tune_object = best_OOD_val_acc
    else:
        raise NotImplementedError('No such senario')
    return tune_object

# 设定optuna的调参范围和优化目标；
def objective(trial):
    """ lr & wd """
    args.learning_rate = trial.suggest_categorical('learning_rate', [0.00001, 0.00005, 0.0001, 0.0005, 0.001])
    args.weight_decay = trial.suggest_categorical('weight_decay', [0.00001, 0.00005, 0.0001])
    
    # our new loss: \sum -log(lii)+ \sum -log(lji)*Wji, i=1, ...,N, (j,i)\in U+
    if args.theroy_view:
        
        # set dataset-specific hyper-params for our new-loss
        if args.dataset == 'PubMed' :
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [300, 500, 700, 1000, 1200, 1400])
            args.update_interval = trial.suggest_categorical('update_interval', [50, 100, 200])
        elif args.dataset == 'WikiCS':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [100, 200, 300, 400, 500])
            args.update_interval = trial.suggest_categorical('update_interval', [50, 100, 200])
        elif args.dataset == 'Coauthor_Physics':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [200, 400, 600, 800, 1000])
            args.update_interval = trial.suggest_categorical('update_interval', [50, 100, 200])
        elif args.dataset == 'Cora':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [50, 100, 150])
            args.update_interval = trial.suggest_categorical('update_interval', [10, 20, 50])
        elif args.dataset == 'CiteSeer':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [100, 200, 300, 400, 500])
            args.update_interval = trial.suggest_categorical('update_interval', [20, 50, 100])
        elif args.dataset == 'GOODCBAS':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [50, 100, 300])
            args.update_interval = trial.suggest_categorical('update_interval', [10, 20, 50, 100])
        elif args.dataset == 'GOODCora' or args.dataset == 'GOODTwitch':
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [100, 300, 500, 700])
            args.update_interval = trial.suggest_categorical('update_interval', [50, 100, 200])
        else:
            args.start_debias_epoch = trial.suggest_categorical('start_debias_epoch', [100, 150, 200, 250, 300])
            args.update_interval = trial.suggest_categorical('update_interval', [20,  40, 80, 100])
        
        if args.llm_emb:
            args.R_W_threshold = trial.suggest_float('R_W_threshold', 0.85, 0.99, step=0.01)
            args.B_W_threshold = trial.suggest_float('B_W_threshold', 0.85, 0.99, step=0.01)
        else:
            # thresholds: 
            args.R_W_threshold = trial.suggest_float('R_W_threshold', 0.8, 0.95, step=0.05)
            args.B_W_threshold = trial.suggest_float('B_W_threshold', 0.8, 0.95, step=0.05)
    
        args.beta = trial.suggest_categorical('beta', [0.1, 0.05, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001])

    # PU density-ratio estimation debias
    if args.PU:
        if not args.sim_as_score: # 用表示的相似度矩阵做为Score_Matrix时是不需要挑这些参数的
            args.PU_num_dim = trial.suggest_categorical('PU_num_dim', [32, 64, 128])
            args.PU_epoch = trial.suggest_categorical('PU_epoch', [300, 400, 499])
            args.PU_alpha = trial.suggest_categorical('PU_alpha', [0, 0.2, 0.4, 0.6, 0.8])
            args.kernel_num = trial.suggest_categorical('kernel_num', [100, 500, 800, 1000])
            args.num_sampled_U = trial.suggest_categorical('num_sampled_U', [3, 5, 10])
        else:
            args.start_PU_epoch = trial.suggest_categorical('start_PU_epoch', [50, 100, 150, 200, 250, 300, 350, 400])
        args.pi = trial.suggest_categorical('pi', [0.001, 0.005, 0.01, 0.05, 0.1])
        if args.num_threshold == 2:
            args.upper_percent = trial.suggest_categorical('upper_percent', [0.7, 0.8, 0.9])
            args.lower_percent = trial.suggest_categorical('lower_percent', [0.3, 0.2, 0.1])
    
    # NaP
    if args.add_pos_pair != 0:
            args.rv_neg_ratio = trial.suggest_float('rv_neg_ratio', 0.05, 0.80, step=0.05)
            args.start_rv_epoch = trial.suggest_int('start_rv_epoch', 10, 300, step=10)
   
   
   
    # set the model-specific hyper-params 
    if args.model == 'GRACE':
        args.tau = trial.suggest_float('tau', 0.1, 0.9, step=0.1)
        args.drop_edge_rate_1 = trial.suggest_float('drop_edge_rate_1', 0.1, 0.6, step=0.1)
        args.drop_edge_rate_2 = trial.suggest_float('drop_edge_rate_2', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_1 = trial.suggest_float('drop_feature_rate_1', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_2 = trial.suggest_float('drop_feature_rate_2', 0.1, 0.6, step=0.1)
        args.gradient_drop_threshold = 1e-3
    elif args.model == 'GCA':
        args.tau = trial.suggest_float('tau', 0.1, 0.9, step=0.1)
        args.drop_edge_rate_1 = trial.suggest_float('drop_edge_rate_1', 0.1, 0.6, step=0.1)
        args.drop_edge_rate_2 = trial.suggest_float('drop_edge_rate_2', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_1 = trial.suggest_float('drop_feature_rate_1', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_2 = trial.suggest_float('drop_feature_rate_2', 0.1, 0.6, step=0.1)
        args.drop_scheme = trial.suggest_categorical('drop_scheme', ['uniform', 'degree', 'evc', 'pr'])
        args.gradient_drop_threshold = 1e-3
    elif args.model == 'BGRL':
        args.drop_edge_rate_1 = trial.suggest_float('drop_edge_rate_1', 0.1, 0.6, step=0.1)
        args.drop_edge_rate_2 = trial.suggest_float('drop_edge_rate_2', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_1 = trial.suggest_float('drop_feature_rate_1', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_2 = trial.suggest_float('drop_feature_rate_2', 0.1, 0.6, step=0.1)
        args.gradient_drop_threshold = 1e-4
    elif args.model == 'COSTA':
        args.costa_k_ratio = trial.suggest_float('costa_k_ratio', 0.1, 0.8, step=0.1)
        args.tau = trial.suggest_float('tau', 0.1, 0.5, step=0.1)
        args.drop_edge_rate_1 = trial.suggest_float('drop_edge_rate_1', 0.1, 0.6, step=0.1)
        args.drop_edge_rate_2 = trial.suggest_float('drop_edge_rate_2', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_1 = trial.suggest_float('drop_feature_rate_1', 0.1, 0.6, step=0.1)
        args.drop_feature_rate_2 = trial.suggest_float('drop_feature_rate_2', 0.1, 0.6, step=0.1)
        args.gradient_drop_threshold = 1e-3
    elif args.model == 'MVGRL':
        args.mvgrl_alpha = trial.suggest_float('mvgrl_alpha', 0.1, 0.9, step=0.1)
        args.gradient_drop_threshold = 1e-5
        args.tolerance_epoch_num = 20
        args.learning_rate = 0.001
    elif args.model == 'DGI':
        args.gradient_drop_threshold = 1e-4
    elif args.model == 'GBT':
        args.p_x = trial.suggest_float('p_x', 0.1, 0.9, step=0.1)
        args.p_e = trial.suggest_float('p_e', 0.1, 0.9, step=0.1)
        args.gradient_drop_threshold = 1e-3
        args.weight_decay = 0.0
        args.num_hidden = 512
        args.batch_size = 4
    # pre-train + val & test
    val_acc = repeat_n_times_sft(_dataset, args, device, trial)
    return val_acc

def pre_process(_dataset, device ,args):
    """
    pre-processing the dataset
    """
    if args.llm_emb:
        args.num_feature = _dataset[0].x.shape[1]
    else:
        args.num_feature = _dataset.num_features
    
    # re-arrange-mask
    if args.dataset == 'WikiCS':
        _dataset._data.val_mask = _dataset[0].val_mask + _dataset[0].stopping_mask
    
    # Giant feat
    # elif args.dataset == 'PubMed':
    #     if args.giant_feat:
    #         giant_xrt_emb = np.load('datasets/PubMed_GIANT_emb/X.all.xrt-emb.npy')
    #         _dataset._data.x = torch.tensor(giant_xrt_emb)
    #         args.num_feature = _dataset.num_features
    #     elif args.tape_feat:
    #         if args.use_gpt:
    #             tape_usegpt_emb = torch.load('datasets/PubMed_TAPE_useGPT_emb/pubmed_use_gpt_torch_emb.pt')
    #             _dataset._data.x = tape_usegpt_emb
    #             args.num_feature = _dataset.num_features
    #     elif args.bert_feat:
    #         bert_emb = torch.load(f'datasets/PubMed_Origin_Bert/PubMed-TA_emb-bert-base-{args.bert_version}.pt')
    #         _dataset._data.x = bert_emb
    #         args.num_feature = _dataset.num_features
    #     else:
    #         raise NotImplementedError 
    
    elif args.dataset == 'ogbn-arxiv' and args.giant_feat:
        giant_xrt_emb = np.load('datasets/ogbn_arxiv_GIANT_emb/X.all.xrt-emb.npy')
        _dataset._data.x = torch.tensor(giant_xrt_emb)
        args.num_feature = _dataset.num_features
    
    if args.model == 'MVGRL':
        pre_processed_path = f'preprocess_res/{args.dataset}' 
        if args.dataset == 'Facebook100':
            target_domain = args.test_dataset if args.test_mode else args.valid_dataset
            if len(args.pre_train_datasets) != 0:
                pre_processed_path = f'preprocess_res/{args.dataset}_small9_{target_domain}'
            else:
                pre_processed_path = f'preprocess_res/{args.dataset}_{target_domain}' 
        if os.path.exists(f'{pre_processed_path}/diff.npy'):
            # print('pre-processed, loading...')
            diff = np.load(f'{pre_processed_path}/diff.npy')
            adj = np.load(f'{pre_processed_path}/adj.npy')
            # print('loaded')
        else:
            # print(f"pre-processing with diffusion, alpha={args.mvgrl_alpha}")
            def diffusion(a: np.ndarray, alpha: float, self_loop: bool =True):
                    if self_loop:
                        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
                    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
                    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
                    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
                    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))
            adj = to_scipy_sparse_matrix(_dataset._data.edge_index).toarray()
            diff = diffusion(adj, alpha=args.mvgrl_alpha)
            if not os.path.exists(f'{pre_processed_path}'):
                os.makedirs(f'{pre_processed_path}')
            with open(f'{pre_processed_path}/diff.npy', 'wb') as f:
                np.save(f, diff)
            with open(f'{pre_processed_path}/adj.npy', 'wb') as f:
                np.save(f, adj)
            # print(f"diffusion pre-processing done, alpha={args.mvgrl_alpha}")
        import scipy.sparse as sparse
        diff_coo = sparse.coo_matrix(diff)
        from torch_geometric.utils.convert import from_scipy_sparse_matrix
        diff_edge_index = from_scipy_sparse_matrix(diff_coo)[0]
        
        _dataset._data.diff_edge_index = diff_edge_index
        _dataset._data.diff = diff
        _dataset._data.adj = adj
    
    elif args.model == 'GCA':
        edge_index = _dataset._data.edge_index
        x = _dataset._data.x
        data = _dataset._data
        # if args.drop_scheme in ['pr', 'degree', 'evc']:
        if args.drop_scheme == 'pr':
            node_pr = compute_pr(edge_index)
            feature_weights = feature_drop_weights(x, node_c=node_pr).to(device)
        elif args.drop_scheme == 'degree':
            edge_index_ = to_undirected(edge_index)
            node_deg = degree(edge_index_[1])
            feature_weights = feature_drop_weights(x, node_c=node_deg).to(device)
        elif args.drop_scheme == 'evc':
            node_evc = eigenvector_centrality(data)
            feature_weights = feature_drop_weights(x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        
        if args.drop_scheme == 'degree':
            drop_weights = degree_drop_weights(edge_index).to(device)
        elif args.drop_scheme == 'pr':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(device)
        elif args.drop_scheme == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None
        _dataset._data.drop_weights = drop_weights
        _dataset._data.feature_weights = feature_weights    

def get_dataset(args):
    "Get the pyg datasets"
    if args.dataset == 'Facebook100':
        from datasets.Facebook100 import load_fb_dataset
        _dataset = load_fb_dataset(args)
    elif args.dataset == 'GOODWebKB':
        from datasets.GOOD_webkb import GOODWebKB
        _dataset = GOODWebKB(root='datasets', domain='university', generate=True, shift='covariate')
    elif args.dataset == 'GOODCora':
        from datasets.GOOD_cora import GOODCora
        domain_flag = args.domain_flag # 'degree' or 'word'
        _dataset = GOODCora(root='datasets', domain=domain_flag, generate=True, shift='covariate')
    elif args.dataset == 'GOODTwitch':
        from datasets.GOOD_twitch import GOODTwitch
        _dataset = GOODTwitch(root='datasets', domain='language', generate=True, shift='covariate')
    elif args.dataset == 'GOODArxiv':
        from datasets.GOOD_arxiv import GOODArxiv
        _dataset = GOODArxiv(root='datasets', domain='time', generate=True, shift='covariate')
    elif args.dataset == 'GOODCBAS':
        from datasets.GOOD_cbas import GOODCBAS
        _dataset = GOODCBAS(root='datasets', domain='color', generate=True, shift='covariate')
    elif args.dataset in ['Cora', "CiteSeer", "PubMed"]: # 不考察OOD
        if args.llm_emb:
            data = torch.load(f'/data1/lvnuoyan/minilmdata/{args.dataset.lower()}/processed/geometric_data_processed.pt')[0]
            # llm-emb as feature (x)
            llm_emb = torch.load(f'/home/wangzixu/DisShiftGRACE/datasets/TAG_LLM_emb/{args.dataset.lower()}_{args.llm_name}_emb.pt')
            data.x = llm_emb
            #  trans to undirected
            from torch_geometric.utils import is_undirected, to_undirected
            if not is_undirected(data.edge_index):
                data.edge_index = to_undirected(data.edge_index)
            _dataset = MyDataset(data)
        else:
            _dataset = Planetoid(root='datasets', name=args.dataset)
    elif args.dataset == 'WikiCS': # treat the stopping loss as val too
        if args.llm_emb:
            data = torch.load(f'/data1/lvnuoyan/minilmdata/{args.dataset.lower()}/processed/geometric_data_processed.pt')[0]
            # 没有提供raw-text, 直接使用TSTAG的LLM处理后的特征即可
            data.x = data.node_text_feat
            #  trans to undirected
            from torch_geometric.utils import is_undirected, to_undirected
            if not is_undirected(data.edge_index):
                data.edge_index = to_undirected(data.edge_index)
            _dataset = MyDataset(data)
        else:
            _dataset = WikiCS(root='datasets/WikiCS')
    elif args.dataset in ['Computers', 'Photo']:
        _dataset = Amazon(root='datasets', name=args.dataset)
    elif args.dataset == 'Coauthor_CS':
        _dataset = Coauthor(root='datasets', name='CS')
    elif args.dataset == 'Coauthor_Physics':
        _dataset = Coauthor(root='datasets', name='Physics')
    elif args.dataset == 'ogbn-arxiv':
        _dataset = PygNodePropPredDataset(name=args.dataset, root='datasets')
    else:
        raise NotImplementedError
    return _dataset

if __name__ == '__main__':
    "Read args"
    args = get_args()
   
    "Select the GPU"
    assert args.gpu_id in range(0, 8)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id) 
    
    "Load dataset"
    _dataset = get_dataset(args)
    
    "Select the device"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    "pre-processing for mvgrl and gca"
    pre_process(_dataset, device, args)
    
    "analysis UPos samples"
    if args.analysis_upos:
        import numpy as np
        
        if args.dataset.startswith('GOOD'):
            data = _dataset[0]
            ID_mask = data.PT_mask + data.SFT_mask + data.SFT_test_mask
            y = np.array(_dataset[0].y.squeeze().to(torch.int64).cpu()[ID_mask])
            same_label_matrix = torch.tensor((y[:, None] == y).astype(int)).to(device)
            import torch.nn.functional as F
            sup_emb = torch.load(f'SupEmb/{args.dataset}_sup_emb.pt').to(device)[ID_mask]
            sup_emb_sim = F.normalize(sup_emb) @ F.normalize(sup_emb).T
        else:   
            y = np.array(_dataset[0].y.squeeze().to(torch.int64).cpu())
            same_label_matrix = torch.tensor((y[:, None] == y).astype(int)).to(device)
            import torch.nn.functional as F
            sup_emb = torch.load(f'SupEmb/{args.dataset}_sup_emb.pt').to(device)
            sup_emb_sim = F.normalize(sup_emb) @ F.normalize(sup_emb).T
        
        epochs_list = []
        if args.dataset == 'Cora':
            epochs_list = [_*20 + 150 for _ in range(19)] if args.model == 'GCA' else [270, 290, 310, 330, 350] #GRACE+ and GCA+
        elif args.dataset == 'GOODTwitch':
            epochs_list = [300 + 100*i for i in range(5)] if args.model == 'GCA' else [300 + 100*i for i in range(5)] #GRACE+ and GCA+
        else:
            raise NotImplementedError
        for e in epochs_list:
            upos_B_W = torch.load(f'UPos/UPos_{args.dataset}_{args.model}_{e}_B_W.pt').to(device)
            upos = upos_B_W
            upos[upos != 0] =1
            upos -= torch.eye(upos.shape[0]).to(device)
            print(f'{e} epoch, same_class_ratio: {(upos*same_label_matrix).sum() / upos.sum()}')
            print(f'{e} epoch, sup_emb_sim: {(upos*sup_emb_sim).sum() / upos.sum()}')
        sys.exit()
            
        # llm_emb = _dataset[0].x.to(device)
        # llm_emb_sim = F.normalize(llm_emb) @ F.normalize(llm_emb).T
        # for e in [150, 170, 190, 210]:
        #     upos_B_W = torch.load(f'UPos_Cora_Llama3.2-1B_GRACE_{e}_B_W.pt').to(device)
        #     upos = upos_B_W
        #     upos[upos != 0] =1
        #     upos -= torch.eye(upos.shape[0]).to(device)
        #     print(f'{e} epoch, llm_emb_sim: {(upos*llm_emb_sim).sum() / upos.sum()}')
    
    "Tune Or Just Run The Pipline"
    if args.tune_hyper_params:
        study = optuna.create_study(direction="maximize", sampler= TPESampler(seed=666))
        study.optimize(objective, n_trials=args.n_trials if args.model== 'GRACE' or 'BGRL' or 'COSTA' else 100)
        
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        
        print("  Number: ", trial.number)
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        repeat_n_times_sft(_dataset = _dataset, args=args, device=device)
    print('\n')

    
    