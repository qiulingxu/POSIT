import os
from datetime import datetime
from tqdm import tqdm
import torch as T
import numpy as np

from model.ease import EASE
from dataset.common import MovieLenDataset, covert_csr_to_torch, covert_csr_to_torch_coo



def main(method, loss_name, lamda, dataset="ml20m", adv_loss = "recall", ones_aware= True, lr = 5e-2, alr=None, adecay=None, worst_ratio=None, adv_struct="linear", s3=True, nstd=1.0, ipwalpha=1.0, weight_decay=1e-5):
    global model, es, name
    if dataset in ["ml20m", "nflx"] :
        batch_size = 1024
        epochs = 50
    else:
        batch_size= 8192
        epochs = 50
    train_loader = MovieLenDataset(dataset, batch_size=batch_size,shuffle=True)
    test_tr, test_te, valid_tr, valid_te = train_loader.test_tr, train_loader.test_te, train_loader.valid_tr, train_loader.valid_te
    n_dim = train_loader.max_item
    print(len(train_loader))


    assert method in ["baseline", "baseline_wl2", "tail", "itemarl", \
        "baseline_wl2_xent", "baseline_xent", "itemarl_ld","ease_ed","ease_ed_xent", "tail_ld",\
            "itemarl_metric", "ipw", "itemarl_metric_tail"]

    if dataset in ["ml20m", "nflx"]:
        lr = 5e-2
    else:
        lr = 8e-2
    if method in ["baseline", "baseline_wl2", "baseline_wl2_xent", "baseline_xent"]:
        name = "{}_ease_lr{:.2e}_wd{:.2e}_{}".format(dataset,lr, weight_decay, method)
    elif method in ["ease_ed", "ease_ed_xent"]:
        hidden = 128
        name = "{}_ease_ed_lr{:.2e}_wd{:.2e}_hn_{}".format(dataset,lr, weight_decay, hidden, method)
    elif method in ["tail", "tail_ld" ]:
        name = "{}_ease_lr{:.2e}_wd{:.2e}_wr{:.2f}_lam{:.2e}_{}_{}".format(dataset,lr, weight_decay, worst_ratio, lamda, loss_name, method)
    elif method in ["itemarl", "itemarl_ld"]:
        name = "{}_ease_lr{:.2e}_wd{:.2e}_awd{:2e}_ast{}_lam{:.2e}_{}_{}".format(dataset,lr, weight_decay, adecay, adv_struct, lamda, loss_name, method)
    elif method in ["ipw" ]:
        name = "{}_ease_lr{:.2e}_wd{:.2e}_alpha{:.2e}_lam{:.2e}_{}_{}".format(dataset,lr, weight_decay, ipwalpha, lamda, loss_name, method)   
    elif method in ["itemarl_metric"]:
        #adv_loss = "recall"
        name = "{}_ease_lr{:.2e}_alr{:.2e}_wd{:.2e}_nstd{:.2e}_awd{:.2e}_ast{}_aloss{}_ones{}_lam{:.2e}_{}_{}".format(dataset,lr, alr, weight_decay, nstd, adecay, adv_struct, adv_loss, ones_aware, lamda, loss_name, method)
    elif method in ["itemarl_metric_tail"]:
        #adv_loss = "recall"
        name = "{}_ease_lr{:.2e}_alr{:.2e}_wd{:.2e}_nstd{:.2e}_awd{:.2e}_ast{}_aloss{}_ones{}_wr{:.2e}_lam{:.2e}_{}_{}".format(dataset,lr, alr, weight_decay, nstd, adecay, adv_struct, adv_loss, ones_aware, worst_ratio, lamda, loss_name, method)        
    else:
        assert False
    
    # Precheck
    s = os.listdir("measure")
    for file in s:
        if file.find(name)>=0:
            print("Duplicate Exp")
            return 0

    if method == "tail":
        from model.tail_opt import Tail
        model = EASE(n_dim = n_dim, eps =0.)# 1.0/np.sqrt(n_dim))
        model_pred = model
        org_model = model
        #dim = 0 for item level
        model = Tail(learner = org_model, loss= loss_name, beta=worst_ratio, leakiness=0., lamda=lamda,  alphascale=1e4)
        opt = T.optim.SGD([{"params": model.learner.parameters(), "lr": lr, "weight_decay":weight_decay, "momentum":0.9},
                            {"params": [model.alpha, ], "lr": 1e-1},], lr=lr)
    elif method == "itemarl":
        from model.aritem_common import ARItem
        model = ARItem(u_num = train_loader.max_user, i_num = train_loader.max_item, lamda=lamda, adv_struct=adv_struct, loss=loss_name)
        opt = T.optim.SGD([{"params": model.learner.parameters(), "lr": lr, "weight_decay":weight_decay, "momentum":0.9},
                            {"params": model.adversary.parameters(), "lr": alr, "weight_decay":adecay},], lr=lr)    
        model_pred = model.learner
        org_model = model.learner
    elif method == "ipw":
        from model.ipw import IPWItem
        model = IPWItem(u_num = train_loader.max_user, i_num = train_loader.max_item, loss=loss_name, lamda=lamda, alpha=ipwalpha)
        opt = T.optim.SGD([{"params": model.learner.parameters(), "lr": lr, "weight_decay":weight_decay, "momentum":0.9}], lr=lr)    
        model_pred = model.learner
        org_model = model.learner

    elif method == "itemarl_metric":
        from model.aritem_recall1s_common import ARItem
        model = ARItem(u_num = train_loader.max_user, i_num = train_loader.max_item, lamda=lamda, moment=0.9, loss=loss_name, adv_loss=adv_loss, ones_aware=ones_aware, adv_struct=adv_struct, nstd=nstd)
        opt = T.optim.SGD([{"params": model.learner.parameters(), "lr": lr, "weight_decay":weight_decay, "momentum":0.9},
                            {"params": model.adversary.parameters(), "lr": alr, "weight_decay":adecay},
                        ], lr=lr)    
        model_pred = model.learner
        org_model = model.learner   
    elif method in ["baseline", ]:
        model = EASE(n_dim = n_dim, eps =0.)# 1.0/np.sqrt(n_dim))
        model_pred = model
        org_model = model
        opt = T.optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        assert False
    #scheduler = T.optim.lr_scheduler.MultiStepLR(opt,milestones=[50,70],gamma=0.2)
    scheduler = T.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,)
    if not T.cuda.is_available():#dataset=="msd":
        device="cpu"
        #model = T.nn.DataParallel(model)
    else:
        device="cuda:0"
    model = model.to(device)
    # Block 3


    from evaluate.fairness import dcg_recall_common_delete
    from evaluate.earlystop import early_stop
    def evaluate_func(model, test_tr, test_te):

        bs = 1000
        cnt = 0 
        dcg_tot = 0
        num = int(test_te.shape[0])//bs
        for i in range(num):
            x = T.tensor(test_tr[i*bs:(i+1)*bs]).to(device)
            y= T.tensor(test_te[i*bs:(i+1)*bs]).to(device) 
            pred = model_pred(x)
            dcg = dcg_recall_common_delete(y, pred, k=100)
            num = int(x.shape[0])
            dcg_tot += dcg * num
            cnt += num
        return (dcg_tot/cnt).item()


    # Block 4

    model = model.train()
    from dataset.common import forever_iter
    from model.ease import sparse_dense_mul
    neg_weight =1.0
    es = early_stop(max_tolerate=10,model=model, optimizer=opt)
    
    more_desc = ""
    if method in ["itemarl_metric", "itemarl_metric_tail"]:
        item = covert_csr_to_torch(train_loader.train_original.transpose()).to(device)
        print(item.shape)
        #iter_on_item = forever_iter(train_loader.iter_on_item)
        
        #model.estimate_step(item)
        for i in range(epochs):
            tbar = tqdm(train_loader)
            for X_user, user_id in tbar:
                #print(user_id[:10])
                opt.zero_grad()
                X_user = X_user.to(device)
                #print("get data")
                loss = model.learner_step(X_user, user_id, scale=1000)
                loss.backward()

                aloss = model.adversary_step_full(item)
                aloss.backward()

                opt.step()
                tbar.set_description("adv loss {} natloss {} ".format(aloss.item(), loss.item() ))
            loss_val = evaluate_func(model, valid_tr, valid_te)
            es.add_loss(loss_val)
            loss_test = evaluate_func(model, test_tr, test_te)
            print("ndcg", loss_val, "on test", loss_test)
            if es.should_stop():
                break
            scheduler.step()
    elif method == "ipw":
        item = covert_csr_to_torch_coo(train_loader.train_original.transpose()).to(device)
        print(item.shape)
        #iter_on_item = forever_iter(train_loader.iter_on_item)
        model.check_count(item)
        #model.estimate_step(item)
        for i in range(epochs):
            tbar = tqdm(train_loader)
            for X_user, user_id in tbar:
                #print(user_id[:10])
                opt.zero_grad()
                X_user = X_user.to(device)
                
                loss = model.learner_step(X_user, user_id, scale=1000)
                loss.backward()

                opt.step()
                tbar.set_description(" natloss {} ".format( loss.item() ))
            loss_val = evaluate_func(model, valid_tr, valid_te)
            es.add_loss(loss_val)
            loss_test = evaluate_func(model, test_tr, test_te)
            print("ndcg", loss_val, "on test", loss_test)
            if es.should_stop():
                break
            scheduler.step()
    else:
        for i in range(epochs):
            tbar = tqdm(train_loader)
            for data_user, index_user in tbar:
                X = data_user.to(device)
                #feat = model._convert_index_to_one_hot(feat)
                opt.zero_grad()
                #feat = T.tensor(feat).to(device)
                #weight = ((T.ones(X.shape).to(device)-X) *neg_weight) + X
                Y = model(X)
                X = X.to_dense()
                if method in ["baseline",]:
                    loss = T.mean(T.square(Y-X) ) * 1e3  #*weight
                elif method == "tail":
                    loss = T.mean(Y)*1e3
                    more_desc = "alpha: " + str(model.alpha.item())
                else:
                    assert False
                #loss = T.mean(T.square(model(X)-X) ) * 1e3
                #feat *= feat_weight
                #weight= T.tensor(weight)
                #loss_adv, loss_normal = model.combo_step(feat, label, weight= weight, scale = 1000 /2) 
                loss.backward()
                opt.step()
                tbar.set_description("loss {} ".format(loss.item() )+ more_desc)
            loss_val = evaluate_func(model, valid_tr, valid_te)
            es.add_loss(loss_val)
            loss_test = evaluate_func(model, test_tr, test_te)
            print("ndcg", loss_val, "on test", loss_test)
            if es.should_stop():
                break
            scheduler.step()
            
        
    model = model.eval()


    model.load_state_dict(es.get_best()["model_state_dict"])
    evaluate(name+ "_best", org_model , test_tr, test_te, n_dim)

def evaluate(model_name, model, test_tr, test_te, item_dim):

    from evaluate.fairness import group_func_by_map_on_npy, FMeasure_Category, utility_per_nrrank, \
                            utility_per_dcg, mrr_common, normalized_dcg_common, recall_common, coverage
    from evaluate.fairness import Coverage_k, Coverage_k_per_m, normalized_dcg_common, mrr_common, recall_common
    from evaluate.fairness import mean_std, mean_stderr, merge_xent, calc_xent, calc_l2
    from recsys_metrics import dcg_ind, recall_ind, normalized_dcg, recall, hit_rate, recall_ind_dist
    import torch as T
    import torch.nn as nn
    from utils.data import NestDict
    import time


    #time.sleep(3600*4)
    device = "cpu"
    global_models = {model_name: model.to(device)}

    now = datetime.now()
    #g_funcs = {"year": group_func_by_map_on_npy("../ml20m_year_tag.npy"),
    #        "genre": group_func_by_map_on_npy(../ml20m_genre_tag.npy"),
    #        "item_id":lambda x: np.arange(x)}
    report = NestDict()
    rst = NestDict()

    mse = nn.MSELoss(reduction="none")
    losses = {
        "xent":nn.BCEWithLogitsLoss(reduction="none"),
        "l2":lambda pred,label: -mse(pred,label),
        "1s":lambda pred,label: label,
        "i_lw_utility": lambda pred,label: utility_per_dcg(pred,label, onlyone=False),
        "i_dcg100": lambda pred,label: dcg_ind(pred, label, k=100),
        "i_recall50": lambda pred, label: recall_ind(pred, label, k=50),
        "i_recall20": lambda pred, label: recall_ind(pred, label, k=20),
        "i_recall100": lambda pred, label: recall_ind(pred, label, k=100),
        "coverage20": lambda pred, label: coverage(pred, label, k=20, u=100),
        "coverage50": lambda pred, label: coverage(pred, label, k=50, u=100),
        "coverage100": lambda pred, label: coverage(pred, label, k=100, u=100),
        "u_ndcg": lambda pred,label: normalized_dcg(pred, label, k=100, reduction="none"),
        "u_recall100": lambda pred,label: recall(pred,label,k=100, reduction="none"),
        "u_recall50": lambda pred,label: recall(pred,label,k=50, reduction="none"),
        "u_recall20": lambda pred,label: recall(pred,label,k=20, reduction="none"),
        "u_hr100": lambda pred,label: hit_rate(pred,label,k=100, reduction="none"),
        "u_hr20": lambda pred,label: hit_rate(pred,label,k=20, reduction="none"),
        "u_hr50": lambda pred,label: hit_rate(pred,label,k=50, reduction="none"),
        "i_recall100_dist": lambda pred,label:recall_ind_dist(pred,label, k=100, reduction="none")
    }

    bs = 400
    cnt = 0 
    tot_num = test_tr.shape[0]
    itemid = np.array(list(range(item_dim)))
    for loss_name in losses.keys():
        rst[loss_name+"_tot"] = 0.
    rst["cnt"]=1e-8
    rst["cnt_pos"] = 1e-8

    preds = []
    with T.no_grad():
        for i in range(tot_num//bs):
            x = T.tensor(test_tr[i*bs:(i+1)*bs]).float().to(device)
            label= T.tensor(test_te[i*bs:(i+1)*bs]).float().to(device) 
            #mask = (label>=0.).float()
            mask = (label>=0.).float()
            mask_pos = (label>0.).float()
            rst["cnt_pos"] += T.sum(mask_pos, dim=0)
            rst["cnt"] += T.sum(mask, dim=0)
            for key in global_models.keys():

                pred = global_models[key](T.tensor(x))
                preds.append(pred.detach().cpu().numpy())
                #print(pred[:10])
                pred = pred.detach().cpu()
                for loss_name in losses.keys():
                    _loss = losses[loss_name](pred, label)
                    if loss_name == "1s":
                        _loss = _loss * mask
                        _loss = T.sum(_loss, dim=0)
                    elif loss_name.find("coverage")>=0:
                        pass
                    elif loss_name.startswith("u_"):
                        _pred = T.where(label<0, T.zeros_like(pred)-1e8, pred)
                        _label = T.where(label<0, T.zeros_like(pred), label)
                        _loss = losses[loss_name](_pred, _label)
                        _loss = T.sum(_loss)
                    else:
                        _loss = _loss *mask_pos
                        _loss = T.sum(_loss, dim=0)
                    
                    rst[loss_name+"_tot"] += _loss
    for loss_name in losses.keys():
        if loss_name == "1s":
            report [10000,model_name, loss_name] = (rst[loss_name+"_tot"]/rst["cnt"]).detach().cpu().numpy()
        elif loss_name.find("coverage")>=0:
            report [10000,model_name, loss_name] = (rst[loss_name+"_tot"]/(tot_num/100))
        elif loss_name.startswith("u_"):
            report [10000,model_name, loss_name] = (rst[loss_name+"_tot"]/tot_num)
        else:
            report [10000,model_name, loss_name + "_pos"] = (rst[loss_name+"_tot"]/rst["cnt_pos"]).detach().cpu().numpy()
            report [10000,model_name, loss_name + "_all"] = (rst[loss_name+"_tot"]/rst["cnt"]).detach().cpu().numpy()

    preds = np.concatenate(preds, axis=0)
    mask = (test_te<0.)
    preds_0 = preds.copy()
    test_te_0 = test_te.copy()
    preds_0[mask] =0.
    test_te_0[mask] =0.
    
    idcg = normalized_dcg(T.tensor(preds_0.transpose()), T.tensor(test_te_0.transpose()), k=None, reduction="none")
    idcg = idcg.detach().numpy()
    report[10000,model_name,"i_ndcg"] = idcg
    report.save("./measure/measure_{}_{}.json".format(model_name, now.strftime("%m-%d-%Y %H-%M-%S")))


if __name__=="__main__":
    main(**{'dataset': 'ml20m', 'method': 'itemarl_metric', 'loss_name': 'l2', 'lr': 0.5, 'adecay': 1e-06, 'lamda': 1.0, 'alr': 0.0001, 'adv_struct': 'linear', 'nstd': 1.0, 'adv_loss': 'freq', 'ones_aware': False, 'ipwalpha': None, 'worst_ratio': 0.5, 'weight_decay': 1e-06})