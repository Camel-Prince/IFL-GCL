import numpy as np
from model import GRACE, DGI
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import InMemoryDataset

def get_all_embeddings(pt_model: torch.nn.Module, dataset: InMemoryDataset, device: torch.device, args):
    "获取全量节点的embedding：z"
    data = dataset._data
    features = data.x.to(device).to(dtype=torch.float32)
    edge_index = data.edge_index.to(device)
    if args.model == 'GRACE' or args.model == 'COSTA' or args.model == 'GCA' or args.model == 'GBT':
        z = pt_model(features, edge_index)
    elif args.model == 'DGI':
        z, _ = pt_model.embed(features, edge_index)
    elif args.model == 'MVGRL':
        features = torch.FloatTensor(features.cpu().numpy()[np.newaxis])
        adj = torch.FloatTensor(data.adj[np.newaxis])
        diff = torch.FloatTensor(data.diff[np.newaxis])
        features = features.to(device)
        adj = adj.to(device)
        diff = diff.to(device)
        z, _ = pt_model.embed(features, adj, diff, sparse=False, msk=None)
        z = z.squeeze()
    elif args.model == 'BGRL':
        z, _, _ = pt_model(features, features, edge_index, edge_index)   
    else:
        raise NotImplementedError
    return z

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def fine_tune(pre_train_model: torch.nn.Module, dataset: InMemoryDataset, device: torch.device, args, **kwargs):
    """
    Fine tune after Pre-train;
    :param model: pre-trained model;
    :param downstream_dataset_dict: fine-tune datasets;
    :param device:
    :return:
    ID-val-acc & ID-test-acc
    """
    
    pre_train_model.eval()
    "获取全量节点的embedding z"
    data = dataset._data
    z = get_all_embeddings(pre_train_model, dataset, device, args)
    labels = data.y

    "Pre-process for SFT"
    X = z.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy().reshape((labels.shape[0],))
    class_weight = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weight = {i: item for i, item in enumerate(class_weight)}
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')

    # 划分sft用的训练和测试集
    from sklearn.model_selection import train_test_split
    if args.dataset == 'WikiCS': # train和val已经给出了20种划分，test固定
        cur_repeat = args.cur_repeat
        X_train, X_val, X_test = X[data.train_mask[:, cur_repeat]], X[data.val_mask[:, cur_repeat]], X[data.test_mask]
        y_train, y_val, y_test = Y[data.train_mask[:, cur_repeat]], Y[data.val_mask[:, cur_repeat]], Y[data.test_mask]
    else:
        if args.senario == 'ID':
            # train val test  1 :1 :8 
            # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
            
            # train val test  1 :4 :5 
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8)
        elif args.senario == 'OOD':
            ID_mask = data.PT_mask + data.SFT_mask + data.SFT_test_mask
            X = X[ID_mask]
            Y = Y[ID_mask]
            # train val test  1 :1 :8 
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
    
    logreg = LogisticRegression(solver='liblinear', class_weight=class_weight)
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0,
                       scoring= 'accuracy')
    "SFT and SFT_test"
    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred_prob)
    test_acc = accuracy_score(y_test, y_pred) 
    
    y_pred_prob = clf.predict_proba(X_val)
    y_pred = prob_to_one_hot(y_pred_prob)
    val_acc = accuracy_score(y_val, y_pred)        
    return {
        'val_acc': val_acc,
        'test_acc': test_acc,
        'clf': clf,
    }

def fine_tune_only4test(pre_train_model: torch.nn.Module, dataset: InMemoryDataset, device: torch.device, args, **kwargs):
    """
    Fine tune after Pre-train;
    :param model: pre-trained model;
    :param downstream_dataset_dict: fine-tune datasets;
    :param device:
    :return:
    ID-val-acc & ID-test-acc
    """
    
    pre_train_model.eval()
    "获取全量节点的embedding z"
    data = dataset._data
    z = get_all_embeddings(pre_train_model, dataset, device, args)
    labels = data.y

    "Pre-process for SFT"
    X = z.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy().reshape((labels.shape[0],))
    class_weight = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weight = {i: item for i, item in enumerate(class_weight)}
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')

    # 划分sft用的训练和测试集
    from sklearn.model_selection import train_test_split
    if args.dataset == 'WikiCS': # train和val已经给出了20种划分，test固定
        cur_repeat = args.cur_repeat
        X_train, X_val, X_test = X[data.train_mask[:, cur_repeat]], X[data.val_mask[:, cur_repeat]], X[data.test_mask]
        y_train, y_val, y_test = Y[data.train_mask[:, cur_repeat]], Y[data.val_mask[:, cur_repeat]], Y[data.test_mask]
    else:
        if args.senario == 'ID':
            # train val test  1 :1 :8 
            # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
            
            # train val test  1 :4 :5 
            # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8)
            
            # train val test  1 :0 :9 
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)
        elif args.senario == 'OOD':
            ID_mask = data.PT_mask + data.SFT_mask + data.SFT_test_mask
            X = X[ID_mask]
            Y = Y[ID_mask]
            # train val test  1 :1 :8 
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
    
    logreg = LogisticRegression(solver='liblinear', class_weight=class_weight)
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0,
                       scoring= 'accuracy')
    "SFT and SFT_test"
    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred_prob)
    test_acc = accuracy_score(y_test, y_pred) 
    
    # y_pred_prob = clf.predict_proba(X_val)
    # y_pred = prob_to_one_hot(y_pred_prob)
    # val_acc = accuracy_score(y_val, y_pred)        
    return {
        # 'val_acc': val_acc,
        'test_acc': test_acc,
        'clf': clf,
    }



def Valid_or_Test(pre_train_model: torch.nn.Module, task_head , dataset: InMemoryDataset, device: torch.device, args):
    """
    Test only on test set
    :param pre_train_model:
    :param task_head:
    :param dataset: the whole dataset with PT_mask, SFT_mask, SFT_test_mask, Test_mask
    :param device:
    :return: result dict
    """
    pre_train_model.eval()
    data = dataset._data

    if args.senario == 'OOD':
        Target_mask = data.Test_mask if args.test_mode else data.Valid_mask
    else:
        Target_mask = data.test_mask if args.test_mode else data.val_mask
    # Target_mask = data.SFT_test_mask
    
    "Take the test data from the whole data"
    z = get_all_embeddings(pre_train_model, dataset, device, args)
    labels = data.y
    test_embeddings = z[Target_mask]
    test_labels = data.y[Target_mask]

    "Pre-process for test"
    X = test_embeddings.detach().cpu().numpy()
    Y = test_labels.detach().cpu().numpy().reshape((test_labels.shape[0],))
    # X = z.detach().cpu().numpy()
    # Y = labels.detach().cpu().numpy().reshape((labels.shape[0],))
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')
    
    "Predict and evaluate"
    from eval import prob_to_one_hot
    Y_pred = task_head.predict_proba(X)
    Y_pred = prob_to_one_hot(Y_pred)
    
   
    target_label_space = torch.unique(data.y[Target_mask])
    sft_label_space = torch.unique(data.y[data.SFT_mask if args.senario == 'OOD' else data.train_mask])
    if target_label_space.shape[0] < sft_label_space.shape[0]:
        """
        align with pt_data; 当测试集合的label数少于sft集合时，需要将Y_pred和Y做对齐；Y中缺少的列补为false
        """
        idx_not_in_target = torch.isin(sft_label_space, target_label_space, invert=True)
        diff_labels = sft_label_space[idx_not_in_target]
        false_vec = np.array([False] * Y.shape[0])
        for j in diff_labels.tolist():
            Y = np.insert(Y, j, false_vec, axis=1)
        
    
    micro = f1_score(Y, Y_pred, average="micro", labels=np.unique(Y_pred), zero_division=1)
    # pre_mi = precision_score(Y, Y_pred, average="micro", labels=np.unique(Y_pred), zero_division=1)
    # re_mi = recall_score(Y, Y_pred, average="micro", labels=np.unique(Y_pred), zero_division=1)

    macro = f1_score(Y, Y_pred, average="macro", labels=np.unique(Y_pred), zero_division=1)
    # pre_ma = precision_score(Y, Y_pred, average="macro", labels=np.unique(Y_pred), zero_division=1)
    # re_ma = recall_score(Y, Y_pred, average="macro", labels=np.unique(Y_pred), zero_division=1)

    # f1_none = f1_score(Y, Y_pred, average=None , labels=np.unique(Y_pred), zero_division=1)
    acc = accuracy_score(Y, Y_pred)

    "统计那些样本具体被如何分错"
    Y_pred = np.argmax(Y_pred, axis=1)
    Y = np.argmax(Y, axis=1)
    wrong_idx = np.where(Y_pred != Y)[0]
    wrong_pred = Y_pred[wrong_idx]
    wrong_true = Y[wrong_idx]
    return {
        'f1_micro': micro,
        'f1_macro': macro,
        'acc': acc,
        'wrong_pred': wrong_pred,
        'wrong_true': wrong_true,
    }

def repeat_statistic(sftTest_res_list, valid_res_list, args, **kwargs):
    # emb_inner_dis_list =  kwargs['emb_inner_dis_list']
    print(f"Seed: {args.seed}")
    
    if args.dataset == 'Facebook100':
        print(f"PT,SFT,SFT_test: {args.pre_train_datasets}")
        print(f"Valid: {args.valid_dataset}")
    else:
        print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Num_epochs: {args.num_epochs}")
    print(f"Repeat_times: {args.repeat_times}")
    print(f'Trial_number: {args.trial_number}')
    if args.model == 'GRACE':
        print(f'tau: {args.tau}')
        print(f'drop_edge_rate_1: {args.drop_edge_rate_1}')
        print(f'drop_edge_rate_2: {args.drop_edge_rate_2}')
        print(f'drop_feature_rate_1: {args.drop_feature_rate_1}')
        print(f'drop_feature_rate_2: {args.drop_feature_rate_2}')
    if args.theroy_view:
        print(f'start_debias_epoch: {args.start_debias_epoch}')
        print(f'update_interval: {args.update_interval}')
        print(f'R_W_threshold: {args.R_W_threshold}')
        print(f'B_W_threshold: {args.B_W_threshold}')
    print(f"SFT_test-acc: {np.mean([res['acc'] for res in sftTest_res_list]) * 100:.2f}%±{np.std([res['acc'] for res in sftTest_res_list]) * 100:.2f}%")
    print(f"Valid-acc: {np.mean([res['acc'] for res in valid_res_list]) * 100:.2f}%±{np.std([res['acc'] for res in valid_res_list]) * 100:.2f}%")
