from queue import Queue
from threading import Thread
import numpy as np
import json
import torch as T
#try:
#    from metaflow import S3
#except:
#    print("No S3 function")
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, T.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return super(MyEncoder, self).default(obj)


class NestDict():
    def __init__(self, final_assign=list):
        self.dct={}
        self.final_assign = final_assign
    
    def get_dict(self):
        return self.dct

    def __getitem__(self, keys):
        if isinstance(keys, str) or isinstance(keys, int):
            keys = [keys, ]
        d = self.dct
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        key = keys[-1]
        if key not in d:
            d[key] = self.final_assign()
        d = d[key]
        return d

    def __setitem__(self, keys, value):
        if isinstance(keys, str):
            keys = [keys, ]
        #print("seting value", keys, value)
        d = self.dct
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        key = keys[-1]
        d[key] = value
        return value

    def save(self, path):
        if path.startswith(r"s3://"):
            local_path = "tmp"
        else:
            local_path = path
        with open(local_path,"w") as f:
            f.write(json.dumps(self.dct, cls=MyEncoder))
        if path.startswith(r"s3://"):
            S3().put_files([(path, local_path)])
    
    def load(self, path):
        #print(path)
        with open(path,"r") as f:
            self.dct = json.loads(f.read())
        return self
def inject(plain_iter, q, process):
    g = plain_iter()
    for stuff in g:     
        if process is not None:
            stuff = process(stuff)
        q.put(stuff)
    q.put(None)

def buffer_iter(num_buffer, plain_iter, preprocess):
    q = Queue(num_buffer)
    p = Thread(target=inject, args=(plain_iter,q, preprocess))
    p.start()
    item = q.get()
    #print("start")
    while item is not None:
        yield item
        item =q.get()
    p.join()    

class np_fast_map():
    def __init__(self, d):
        self.mx = max(d.keys())
        self.d_array = np.zeros((self.mx+1,), dtype=np.int16)
        for k,v in d.items():
            self.d_array[k] =v
    def translate(self, x):
        return self.d_array[x]
