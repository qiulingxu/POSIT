from typing import List
import torch as T
import json
import numpy as np
#import tensorflow as tf
import math
from recsys_metrics import normalized_dcg, mean_reciprocal_rank, recall
#from metaflow import S3
try:
    from utils.data import NestDict
    from .metric import Metric
except:
    import sys, os
    dir_name = sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0,dir_name)
    from metric import Metric
    from utils.data import NestDict

def group_func_by_key_factory(key):
    def group_func_by_key(x):
        return x[key].to_numpy()
    return group_func_by_key

def group_func_by_map_on_npy(filepath:str):
    if filepath.startswith(r"s3://"):
        s3 = S3().get(filepath)
        open("tmp.npy","wb").write(s3.blob)
        filepath = "tmp.npy"
    arr = np.load(filepath)
    print(filepath, arr.shape)
    def group_func_by_key(x):
        return arr
    return group_func_by_key

def mean_std(vals):
    return {"mean":np.mean(vals), "std":np.std(vals)}

def mean_stderr(vals):
    return {"mean":np.mean(vals), "stderr":np.std(vals)/len(vals)}


def calc_xent(pred, label):
    pred = np.clip(pred, 0., 1.0)
    xent =  -(label * np.log(np.maximum(pred, 1e-8)) + (1 - label) * np.log(np.maximum(1-pred, 1e-8)))
    #xent_pos = xent[label==1.0]
    #xent_neg = xent[label==0.0]
    #xent_tot  =  (xent_pos.mean() + xent_neg.mean()) /2
    return xent#xent_tot

def calc_l2(pred, label):
    l2 = np.square(pred - label)
    return l2

def merge_xent(xent, label):
    label = np.concatenate(label, axis=0)
    xent = np.concatenate(xent, axis=0)
    xent_pos = xent[label==1.0]
    xent_neg = xent[label==0.0]
    xent_tot  =  (xent_pos.mean() + xent_neg.mean()) /2
    return xent_tot


def utility_func_mse(pred, label):
    return np.square(pred-label)

def profile_decorator(func):
    def profile_metric(pred, profile):
        return func(pred, label = profile["label"].to_numpy(), weight = profile["weight"].to_numpy())
    return profile_metric


def utility_per_nrrank(pred, label, weight=None,  dim=-1, onlyone=True, **karg):
    if weight is None:
        weight = label
    rrank = (1 / (np.argsort(np.argsort(-pred, axis=dim), axis=dim) + 1))
    
    if onlyone:
        rrank = rrank[label == 1.0]
    #sw = -np.sort(-weight.to_numpy())
    #print(rrank.shape, weight.shape, label)
    #print(rrank, len(rrank))
    weight = weight
    if onlyone:    
        weight = weight[label == 1.0]
    #sw = np.ones(shape=(rrank.size))
    #rrank_norm =  rrank / (sw[0:rrank.size] / (np.arange(rrank.size) + 1)).sum()
    return weight / rrank

def utility_per_dcg(pred, label, weight=None,  dim=-1, onlyone=True, **karg):
    if weight is None:
        weight = label
    rrank = (1 / np.log2(np.argsort(np.argsort(-pred, axis=dim), axis=dim) + 2))
    
    if onlyone:
        rrank = rrank[label == 1.0]
    #sw = -np.sort(-weight.to_numpy())
    #sw = np.ones(shape=(rrank.size))
    weight = weight
    if onlyone:    
        weight = weight[label == 1.0]
    #rrank_norm =  rrank / (sw[0:rrank.size] / np.log2(np.arange(rrank.size) + 2)).sum()
    return weight / rrank

def coverage(pred, label, k=100, u=100):
    assert pred.shape[0] % u ==0, "Have to include number of users proportional to u"
    #print(pred)
    topk_ind = np.argsort(-pred, axis=-1)[:,:k]
    #print(topk_ind)
    bs = int(pred.shape[0] / u)
    rst = 0.
    for i in range(bs):
        rst += len(np.unique(topk_ind[i*u:(i+1)*u,:].flatten() ))
    return rst

class Coverage_k(Metric):
    def __init__(self, k):
        self.k = k
    
    def calc_batch(self, pred, itemid, **karg):
        topk_ind = np.reshape(np.argsort(-pred)[:self.k],newshape=[-1])
        return set(itemid[topk_ind])
    
    def save_bucket_type(self):
        return set

    def save_bucket_instance(self):
        return set()

    def _aggregate(self, xs):
        assert isinstance(xs, set)
        return {"coverage": len(xs)}

    def add(self, save_bucket: set, x:set):
        save_bucket = save_bucket.union(x)
        return save_bucket

class Coverage_k_per_m(Metric):
    def __init__(self, k, m=10 ):
        # Coverage of top k items per m users
        self.k = k
        self.m = m
    
    def calc_batch(self, pred, itemid, **karg):
        topk_ind = np.reshape(np.argsort(-pred)[:self.k],newshape=[-1])
        return set(itemid[topk_ind])
    
    def save_bucket_type(self):
        assert False, "not supported"

    def save_bucket_instance(self):
        return {"curr_cov":set(), "tot_lst":[], "cnt":0}

    def _report(self, xs):
        assert isinstance(xs, dict)
        return len(xs["curr_cov"])

    def _aggregate(self, xs):
        assert isinstance(xs, dict)
        if len(xs["tot_lst"]) == 0:
            return {"mean":0, "std": 0 }
        else:
            return mean_std(xs["tot_lst"])
        

    def add(self, save_bucket: dict, x:set):
        save_bucket["cnt"] += 1
        save_bucket["curr_cov"] = save_bucket["curr_cov"].union(x)
        if save_bucket["cnt"] % self.m == 0:
            save_bucket["tot_lst"].append(self._report(save_bucket))
            save_bucket["curr_cov"] = set()
        return save_bucket


def dcg_recall(datax, target_pred, k=100, r1=20, r2=50):
    print('computing dcg and recall\n')
    target_pred = target_pred -100*tf.cast(datax < 0, dtype=tf.float32)
    top_k_pred_sort = tf.math.top_k(target_pred, k=k)
    top_k_label_sort = tf.math.top_k(datax, k=k)

    row_index = tf.reshape(tf.constant([[j for _ in range(0,k)] for j in range(0,datax.shape[0])]), shape=(-1,1))
    col_index = tf.reshape(top_k_pred_sort.indices, shape=(-1,1))
    top_k_pred_sort_labels = tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))

    discount_weights = tf.constant([ math.log(2.0 + j)/math.log(2.0) for j in range(0,k)])
    dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)

    col_index = tf.reshape(top_k_label_sort.indices, shape=(-1,1))
    ideal_dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)
    ind = ideal_dcg_by_row > 0
    
    dcg = tf.reduce_mean(dcg_by_row[ind]/ideal_dcg_by_row[ind])
    dcg_by_row = dcg_by_row[ind]/ideal_dcg_by_row[ind]
    
    recall_1 = tf.reduce_sum(tf.gather(top_k_pred_sort_labels, [_ for _ in range(0,r1)], axis=1), axis=1)
    recall_2 = tf.reduce_sum(tf.gather(top_k_pred_sort_labels, [_ for _ in range(0,r2)], axis=1), axis=1)
    label_sum = tf.reduce_sum(datax*tf.cast(datax >0, dtype=tf.float32), axis=1)
    ind = label_sum > 0
    final_recall_1 = tf.reduce_mean(recall_1[ind]/tf.math.minimum(label_sum[ind], r1))
    final_recall_2 = tf.reduce_mean(recall_2[ind]/tf.math.minimum(label_sum[ind], r2))
    recall_1 = recall_1[ind]/tf.math.minimum(label_sum[ind], r1)
    recall_2 = recall_2[ind]/tf.math.minimum(label_sum[ind], r2)
    return (dcg, final_recall_1, final_recall_2, dcg_by_row, recall_1, recall_2)    


def dcg_recall_common_delete(datax, target_pred, k=100):
    #Deprecated
    with T.no_grad():
        #print('computing dcg and recall\n')
        mask = (datax<0.)
        datax = T.where(datax<0, T.zeros_like(datax), datax)
        target_pred = target_pred -1e8*mask
        """mask = (datax>0.)
        top_rank = T.argsort(T.argsort(-target_pred,dim =-1),dim=-1)
        
        valid_test = datax *mask#) / T.sum(mask)
        dcg = (valid_test/T.log2(top_rank+2))[:,:k]
        dcg = T.sum(dcg,dim=-1)
        valid_test, _ = T.sort(valid_test, dim=-1,descending=True)
        valid_test = valid_test[:,:k]
        #print(valid_test[:2])
        
        ideal_dcg = valid_test / T.log2(T.arange(valid_test.shape[1])+2).reshape((1,-1)).to(datax.device)
        row_ind = (ideal_dcg > 0.)
        ideal_dcg = T.sum(ideal_dcg,dim=-1)
        #print(dcg[:10], ideal_dcg[:10])
        #print(dcg/ideal_dcg)
        return T.mean(dcg/ideal_dcg)
        """
        return normalized_dcg(target_pred, datax, k=k)


def decorator_for_common_dataset(func, ):
    class MeanMetric(Metric):
        def __init__(self, device=None, **karg):
            self.karg = karg
            self.device = device
        
        def calc_batch(self, pred, label, **karg):

            pred =  T.tensor(pred)#.to(self.device)
            label = T.tensor(label)#.to(self.device)

            #label = label -1e8*mask
            ret = func(pred, label, reduction="none", **self.karg)
            if isinstance(ret, T.Tensor):
                ret = ret.cpu().numpy()
            return ret

        def save_bucket_type(self):
            assert False, "not supported"

        def save_bucket_instance(self):
            return []

        def _aggregate(self, xs):
            assert isinstance(xs, list)
            xs = np.concatenate(xs, axis=0)
            if len(xs) == 0:
                return {"mean":0, "std": 0 }
            else:
                return mean_std(xs)    

        def add(self, save_bucket: list, x:np.ndarray):
            save_bucket.append(x)
            return save_bucket

    return MeanMetric

normalized_dcg_common = decorator_for_common_dataset(normalized_dcg)
mrr_common =  decorator_for_common_dataset(mean_reciprocal_rank)
recall_common = decorator_for_common_dataset(recall)

top_coverage_100 = Coverage_k(100)
top_coverage_50 = Coverage_k(50)
top_coverage_20 = Coverage_k(20)
top_coverage_100_per_100 = Coverage_k_per_m(100, 100)
top_coverage_50_per_100 = Coverage_k_per_m(50, 100)
top_coverage_20_per_100 = Coverage_k_per_m(20, 100)

utility_per_nrrank_profile = profile_decorator(utility_per_nrrank)
utility_per_dcg_profile = profile_decorator(utility_per_dcg)


#modify it to class


class FMeasure():
    def __init__(self, group_method, **karg):
        assert group_method in ["percentage", "kernel"]
        if group_method == "percentage":
            self.calc = self.calc_group
            self.percent = karg["percent"]

    def calc_group_2d(self, group, utility, output="wg2a", **karg):
        group = np.concatenate(group, axis=0)
        utility = np.concatenate(utility, axis=0)
        return self.calc_group(group, utility, **karg)
    def calc_group(self, group, utility, output="wg2a", use_loss=False, max=False):
        # wg2a worst group to average 
        group = np.array(group)
        utility = np.array(utility)
        assert len(group) == len(utility)
        if use_loss :
            utility = -utility
        #print(group, utility)
        rank = np.argsort(group)
        group = group[rank]
        utility = utility[rank]
        tot_num = len(group)
        group_num = int(self.percent * tot_num)
        s = 0
        e = group_num
        csum = np.sum(utility[:e])
        ext_sum = csum
        ans = s
        sums = []
        for i in range(e, tot_num):
            csum += utility[i] - utility[i-group_num]
            sums.append(csum/group_num)
            if (not max and csum< ext_sum) or (max and csum>ext_sum):
                ext_sum = csum
                ans_e = i
                ans_s = i-group_num+1 
        sums = np.array(sums)
        if use_loss:
            ext_sum = - ext_sum
            sums = -sums
        return (ext_sum /group_num), np.mean(sums), np.std(sums)

class FMeasure_Category():
    def __init__(self, ):
        pass

    def calc_group_2d(self, group, utility, output="wg2a", **karg):
        group = np.concatenate(group, axis=0)
        utility = np.concatenate(utility, axis=0)
        return self.calc_group(group, utility, **karg)

    def calc_group(self, group, utility, output="wg2a", use_loss=False, max=False):
        # wg2a worst group to average 
        group = np.array(group)
        utility = np.array(utility)
        assert len(group) == len(utility), "{} vs {}".format(len(group),len(utility))
        if use_loss :
            utility = -utility
        #print(group, utility)
        #rank = np.argsort(group)
        #group = group[rank]
        #utility = utility[rank]
        tot_num = len(group)

        cate_tot = {}
        cate_cnt = {} 
        for i in range(tot_num):
            gname = group[i]
            if gname not in cate_tot:
                cate_tot[gname] = 0
                cate_cnt[gname] = 0
            cate_cnt[gname] +=1
            cate_tot[gname] += utility[i]
        ext_sum = None
        tot = []
        for k in cate_cnt.keys():
            # filter out group with too little impact
            #if cate_cnt[k]<tot_num*0.01:
            #    continue
            val = cate_tot[k]/cate_cnt[k]
            if ext_sum is None or (max and val>ext_sum) or (not max and val<ext_sum):
                ext_sum = val
            tot.append(val)

        if use_loss:
            ext_sum = -ext_sum
            tot = -np.array(tot)
        return ext_sum, np.mean(tot), np.std(tot) / np.sqrt(len(tot))


if __name__ == "__main__":

    #a = [np.random.rand(10,) for i in  range(10)]
    #label = [np.round(np.random.rand(10,),decimals=0) for i in  range(10)]
    #xent = [calc_xent(_a ,_b) for (_b, _a) in zip(a,label)]
    #print(merge_xent(xent, label))
    
    nd = NestDict()
    nd[1, "b", "c"] = set([1,2,3])
    nd[12,"B"] = 1
    nd["c"].append(1)
    nd["cdd"].append(1)
    print(nd.dct)
    print(nd["a","b","c"])

    group = np.array([0,1,0,1,0,1,0,1,0,1,0,1])
    utility = np.arange(0,1,step = 1.0/len(group))
    #measure = FMeasure(group_method = "percentage", percent = 0.5)
    measure = FMeasure_Category()
    print(measure.calc_group(group,utility))
    

    nd = NestDict()
    for j in [1,2,3]:
        nd[j, "metric"] = 0.
    for i in range(200):
        for j in [1,2,3]:
            a = np.array(np.random.rand(100, 1000), dtype=np.float32)
            id = np.array(np.random.rand(100, 1000), dtype=np.float32)
            #rst = top_coverage_50_per_100.calc_batch(pred=a, itemid=id)
            nd[j, "metric"] += coverage(a, id, k=100, u=10) #top_coverage_50_per_100.add(nd[j, "metric"], rst)
            #top_coverage_50.aggregate(s)

    for j in [1,2,3]:
        print(nd[j, "metric"]/200/10)

    