import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np


from  sklearn.linear_model import SGDRegressor
from recsys_metrics import recall_ind
# This version we will put linear correction into the factor

import scipy.stats as stat

class Adv_Reshape(nn.Module):
    def __init__(self, dim, i_num, device=None, **karg):
        super().__init__()
        self.i_num = i_num
        vals = []
        self.dim = dim
        for i in range(1,i_num+1):
            vals.append(stat.norm.ppf(float(i)/ (i_num+1)))
        self.register_buffer("gauss_point", T.from_numpy(np.array(vals)))
        self.bn = nn.BatchNorm1d(dim, **karg)
        self.to(device)
    def forward(self, x):
        x = self.bn(x)
        if self.training:
            pass
        else:
            idx = T.argsort(x, dim=0)
            with T.no_grad():
                target_val = T.gather(self.gauss_point.unsqueeze(1).repeat_interleave(repeats=self.dim, dim=1), dim=0, index=idx,)
                delta = (target_val - x).detach()
            x =  x + delta
        return x

class resize(nn.Module):
    def __init__(self, nstd):
        super().__init__()
        self.nstd = nstd
    
    def forward(self,x):
        return x * self.nstd

class ARItem_Adversary(nn.Module):
    def __init__(self, embed_size, lamda, adv_struct:str, device="cpu", nstd=1.0, i_num=None):
        super().__init__()

        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()
        self.embed_size = embed_size
        self.device = device
        self.nstd = nstd
        self. i_est = None
        
        #assert adv_struct in ["linear", "linearbn", "2mlp", "wlinearbn", "blinearbn", "b2mlp"]
        self.adv_struct = adv_struct
        self.norm = lambda x: x
        
        if adv_struct in ["linearbn", "wlinearbn", "blinearbn"]:
            self.embedding = nn.Sequential(
                                        nn.Linear(embed_size, 1, bias=False, device=device),                     
                                        )
            self.embedding[0].weight.data.fill_(0.)
            self.norm = nn.BatchNorm1d(1, affine=False, device=device)
        elif adv_struct in ["linearrs"]:
            self.embedding = nn.Sequential(
                                        nn.Linear(embed_size, 1, bias=False, device=device),                     
                                        )
            self.embedding[0].weight.data.fill_(0.)
            self.norm = Adv_Reshape(1, i_num=i_num, affine=False, device=device)
        elif adv_struct.startswith("2mlprs") :
            if adv_struct.startswith("2mlprs"):
                num = int(adv_struct[6:])
            else:
                num = int(adv_struct[5:])
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device,track_running_stats=False),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        Adv_Reshape(1, i_num=i_num, affine=False, device=device,track_running_stats=False)
                                        )
            #self.embedding.weight.data.fill_(0.)
            #self.norm[1].weight.data.fill_(0.)
        elif adv_struct.startswith("2mlp") or adv_struct.startswith("w2mlp"):
            if adv_struct.startswith("2mlp"):
                num = int(adv_struct[4:])
            else:
                num = int(adv_struct[5:])
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        nn.BatchNorm1d(1, affine=False, device=device)
                                        )
            self.embedding.weight.data.fill_(0.)
        elif adv_struct.startswith("2xmlp"):

            num = int(adv_struct[5:])

            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        )
            self.embedding.weight.data.fill_(0.)            
        elif adv_struct.startswith("2bmlp"):
            if adv_struct.startswith("2bmlp"):
                num = int(adv_struct[5:])
            else:
                assert False
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        resize(nstd),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        nn.BatchNorm1d(1, affine=False, device=device)
                                        )
            self.embedding.weight.data.fill_(0.)
            #self.norm[1].weight.data.fill_(0.)
        elif adv_struct.startswith("3mlp") or adv_struct.startswith("w2mlp"):
            if adv_struct.startswith("3mlp"):
                num = int(adv_struct[4:])
            else:
                num = int(adv_struct[5:])
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        nn.Tanh(),
                                        nn.Linear(num, num, bias=False, device=device),
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        nn.BatchNorm1d(1, affine=False, device=device)
                                        )
            self.embedding.weight.data.fill_(0.)
        elif adv_struct.startswith("3bmlp"):
            if adv_struct.startswith("3bmlp"):
                num = int(adv_struct[5:])
            else:
                num = int(adv_struct[5:])
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        resize(nstd),
                                        nn.Tanh(),
                                        nn.Linear(num, num, bias=False, device=device),
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        resize(nstd),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        nn.BatchNorm1d(1, affine=False, device=device)
                                        )
            self.embedding.weight.data.fill_(0.)
            #self.norm[1].weight.data.fill_(0.)
        elif adv_struct .startswith("b2mlp") :
            num = int(adv_struct[5:])
            self.embedding = nn.Linear(embed_size, num, bias=False, device=device)
            self.norm = nn.Sequential(
                                        nn.BatchNorm1d(num, affine=False, device=device),
                                        nn.Tanh(),
                                        nn.Linear(num, 1, bias=False, device=device),
                                        nn.BatchNorm1d(1, affine=False, device=device)
                                        #nn.Linear(embed_size, 1, bias=True, device=device),
                            )        
        elif adv_struct .find("linear") >=0 :
            self.embedding = nn.Sequential(
                                        nn.Linear(embed_size, 1, bias=True, device=device), 
                                        )
            self.embedding[0].weight.data.fill_(0.)
        else:
            assert False
        #self.embedding.weight.data.fill_(0.)
    
    def train(self, mode=True):
        if self.adv_struct.startswith("2mlprs"):
            self.norm[3].train(mode)
    
    def eval(self):
        if self.adv_struct.startswith("2mlprs"):
            self.norm[3].eval()
        pass

    def forward(self, x: T.Tensor, before_sigmoid=False):
        """
        The forward step for the adversary.
        """
        if self.i_est is None:
            if x.is_sparse:
                s = T.sparse.sum(x,dim=None)
            else:
                s = T.sum(x,)
            num = x.shape[0] * x.shape[1]
            mean  = s / num
            std = T.sqrt((T.square(1.0 - mean) *s  + T.square(mean) *(num-s)) / num)
            self.i_est = (std, mean)
        if self.adv_struct.startswith("w"):
            if x.is_sparse:
                cnt = T.sparse.sum(x,dim=1).to_dense().unsqueeze(1)
            else:
                cnt = T.sum(x, dim=1).unsqueeze(1)
            o = self.embedding(x) / cnt
        elif self.adv_struct.startswith("b"):

            #Normalization to provide negative sample a very small weight
            o_1 = self.embedding(x)
            o_2 = self.embedding((T.zeros(size=(1, x.shape[1])).to(x.device) - self.i_est[1]))
            o = (o_1 - o_2).div_(self.i_est[0]) 
        else:
            o = self.embedding(x)
        o = self.norm(o).squeeze(dim=1) 
        if before_sigmoid:
            return o
        #x = x.to_dense()
        
        #print(o.shape, x.shape)
        o = o * self.nstd #* 2
        o = self.sigmoid(o)
        #o = (self.lamda * o + T.ones_like(o)) /( 1+ self.lamda)
        #print(o)
        return o

    def normalize(self, o):
        o_mean = T.mean(o)
        o = o / (o_mean .detach())#T.max(T.Tensor([o_mean, 1e-8])).detach() #
        #o = (self.lamda * o + T.ones_like(o)) /( 1+ self.lamda)
        return o 
