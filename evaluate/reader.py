
from matplotlib import pyplot as plt
from utils.data import NestDict
import glob
import os
import copy
# https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
# remove anything from the list that is not a file (directories, symlinks)
# thanks to J.F. Sebastion for pointing out that the requirement was a list 
# of files (presumably not including directories)  
def get_file_by_date(*search_dirs):
    files = []

    for search_dir in search_dirs:
        files = files + list(filter(os.path.isfile, glob.glob(search_dir + "*.json")))
        #files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    print(files)
    files.sort(key=lambda x: x[x.rfind("_")+1:], reverse=True)
    return files

def overwrite_nest_dict(origin, new, summerize=True):
    if summerize:
        def get_value(val):
            if isinstance(val, float):
                y = copy.copy(val)
                #print(key, y)
            elif isinstance(val, list):
                y =  copy.copy(sum(val) / len(val))
            else:
                assert False, type(val)
            return y
    else:
        assert False
        def get_value(val):
            return val

    for key, item in new.items():
        if key in origin:
            if isinstance(item, dict):
                overwrite_nest_dict(origin[key], new[key], summerize=summerize)
            else:
                origin[key] = get_value(new[key])
        else:
            origin[key] = get_value(new[key])

from tqdm.notebook import tqdm
def get_merge_rst(files,keywords, exclude=None, summerize=False):
    max_common_id = 10000
    print(max_common_id)
    dup  =set()
    dct = {}
    def simplify(org_dct, dct):
        for key in dct.keys():
            item = dct[key]
            #print(key, type(item))
            if isinstance(item, dict):
                if key not in org_dct:
                    org_dct[key] = {}
                org_dct[key] = simplify(org_dct[key],item)
            elif isinstance(item, list):
                org_dct[key] = sum(item) / len(item)
            elif  isinstance(item, float):
                org_dct[key] = item
            else:
                assert False, type(item)
                
        return org_dct     
    if not isinstance(keywords, list):
        keywords = [keywords]
    for file in tqdm(files):
        f = True
        for keyword in keywords:
            if file.find(keyword) <0:
                f = False
        if not f:
            continue
        if exclude and file.find(exclude)>=0:
            continue
        unique_k = file[:file.rfind("_")]
        if unique_k in  dup:
            continue
        dup.add(unique_k)
        n = NestDict()
        n.load(file)
        dct_file = n.get_dict()
        simplify(dct,dct_file[str(max_common_id)])
        #overwrite_nest_dict(dct, dct_file[str(max_common_id)], summerize= summerize)
    return dct

import pickle    
def get_merge_rst_file(file):
    rst = pickle.loads(open(file, "rb").read())
    r = NestDict()
    r.dct = rst
    return r

def draw_figure_per_metric(dct: dict):
    metric_perf = {}
    for model_name, metrics in dct.items():
        for k, v in metrics.items():
            if k in ["wmrr", ]:
                pass

def search_best(filter, trans_dct, key="u_ndcg"):

    mdct = trans_dct.dct[key]
    max_val = -1e9
    best_model_name = None
    tuples = []
    for model_name in mdct.keys():
        flag = True
        for f in filter:
            if model_name.find(f)<0:
                flag=False
        v = mdct[model_name]
        if flag:
            tuples.append((model_name,v))
            if mdct[model_name]>max_val:
                max_val = v
                best_model_name = model_name
    tuples=sorted(tuples,key=lambda x:x[1],reverse=True)
    return best_model_name, tuples

import pickle

class plt_recorder():
    def __init__(self):
        self.cnt = 0
        self.commands_data = []
    
    def record(self, *arg, **karg):
        self.cnt +=1
        self.commands_data.append((arg, karg))

    def replay(self,):
        self.cnt += 1
        return self.commands_data[self.cnt-1]

    def save(self, name):
        with open(name,"wb") as f:
            pickle.dump(self.commands_data,f)
    
    def load(self, name):
        with open(name,"rb") as f:
            self.commands_data = pickle.load(f)
            self.cnt = len(self.commands_data)

    def decorate(self, f):
        def new_f(*arg, **karg):
            self.record(*arg, **karg)
            f(*arg,**karg)
        return new_f