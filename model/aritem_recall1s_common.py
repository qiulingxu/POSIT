
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np

try:
    from .ease import EASE
except:
    from ease import EASE

from  sklearn.linear_model import SGDRegressor
from recsys_metrics import recall_ind, recall_ind_dist, dcg_ind, wpdcg_ind
# This version we will put linear correction into the factor

from .arlitem_adv import ARItem_Adversary



class ARItem(nn.Module):
    def __init__(self, u_num,  i_num, lamda, loss,  adv_loss, ones_aware, adv_struct="linear", device="cpu", moment=0., **karg):
        super().__init__()
        #self.group_weight = nn.Embedding(self.num_leaves, 1, device=self.device)
        self.u_num = u_num
        self.i_num = i_num
        self.lamda = lamda
        self.loss = loss

        self.register_buffer("item_weight", T.ones((i_num,)))

        self.register_buffer("item_loss", T.zeros((i_num,)) ) # -1e-15
        self.register_buffer("loss_count", T.zeros((i_num,))+ 1e-15)
        #self.popularity_linear_comp = T.nn.Linear(1,1,bias=True)
        #self.popularity_linear_comp.weight.data.fill_(0.)
        #self.popularity_linear_comp.bias.data.fill_(0.)
        if loss == "l2":
            self.learner = EASE(i_num, eps=0.)
        elif loss == "xent":
            self.learner = EASE(i_num, eps=0., xent=True)
        self.adv_loss = adv_loss
        self.adversary = ARItem_Adversary(u_num, lamda=lamda, adv_struct=adv_struct, i_num=i_num, **karg)
        self.learner.to(device)
        self.adversary.to(device)
        self.device = device
        self.moment = moment
        self.ones_aware = ones_aware

    def update_item_loss(self, loss_per_item, counts):
        with T.no_grad():
            self.item_loss = self.item_loss * self.moment + loss_per_item  #+ 1e-8
            self.loss_count = self.loss_count * self.moment + counts # + 1e-15


    def forward(self, x:np.ndarray):
        o = self.learner(x)
        return o    

    def check_count(self, x_items):
        if x_items.is_sparse:
            self.counts = (T.sparse.sum(x_items, dim=1).to_dense())
        else:
            self.counts = (T.sum(x_items, dim=1).to_dense())
        self.counts = self.counts / T.min(self.counts)
        self.alpha = T.exp(-self.alpha*self.counts)
        self.item_weight = self.alpha / T.mean(self.alpha)


    def adversary_step_full(self, x_items):
        self.adversary.zero_grad()
        item_weight = self.adversary(x_items)
        with T.no_grad():
            self.item_weight = item_weight
        nitem_weight = self.adversary.normalize(item_weight)

        loss = self.item_loss /self.loss_count
        loss_mean = T.mean(loss)
        loss_l1 =  T.mean(T.abs(loss))
        loss_norm = ((loss ) / loss_l1).detach()
        tot_loss = nitem_weight * loss_norm
        loss = -T.mean(tot_loss)
        return loss



    def learner_step(self, x_users, user_id, scale):
        self.learner.zero_grad()

        pred = self.learner(x_users)
        x_users = x_users.to_dense()
        if self.loss == "l2":
            loss = T.square(pred - x_users) 
        elif self.loss == "xent":
            Xd = x_users.to_dense()
            loss = F.binary_cross_entropy_with_logits(pred,Xd,reduction="none")
        else:
            assert False
        loss_per_item = T.mean(loss, dim=0) * scale
        if self.adv_loss == "recall":         
            adv_loss_per_item = -T.sum(recall_ind(pred.detach(), x_users, k=100),dim=0)
        elif self.adv_loss == "recall_dist":
            adv_loss_per_item = -T.sum(recall_ind_dist(pred.detach(), x_users, k=100),dim=0)
        elif self.adv_loss == "l2":
            adv_loss_per_item = T.sum(T.square(pred.detach() - x_users) , dim=0) 
        elif self.adv_loss == "xent":
            Xd = x_users
            adv_loss_per_item = T.sum(F.binary_cross_entropy_with_logits(pred.detach(),Xd,reduction="none") , dim=0)
        elif self.adv_loss == "dcg":
            Xd = x_users
            adv_loss_per_item = -T.sum(dcg_ind(pred, Xd, k=100), dim=0)
        elif self.adv_loss == "wexposure":
            adv_loss_per_item = T.sum(wpdcg_ind(pred.detach(),x_users, k=100, reduction="none"), dim=0)
        elif self.adv_loss == "freq":
            adv_loss_per_item = -T.sum(x_users.to_dense(), dim=0)
        else:
            assert False
        
        #

            #self.item_loss = self.moment*self.item_loss  +(1-self.moment)*loss_per_item#
            #.scatter_(dim =0, index = user_id, src=loss_per_item) = self.item_loss.select(dim=0, )
        #select_slice = T.select(
        #select_slice = loss

        #version 4
        """item_weight = self.adversary.normalize(self.item_weight.detach())
        loss = item_weight * loss_per_item

        #version 5
        """
        weight1 = self.adversary.normalize(self.item_weight.detach())
        if self.ones_aware:
            if x_users.is_sparse:
                counts = (T.sparse.sum(x_users, dim=0).to_dense()) #/ x_users.shape[0]
            else:
                counts = T.sum(x_users,dim=0)
            self.update_item_loss(adv_loss_per_item, counts)
        else:
            self.update_item_loss(adv_loss_per_item, T.ones_like(adv_loss_per_item) * x_users.shape[0])
        #print(loss_per_item.shape, weight1.shape)
        f_loss = loss_per_item * weight1
        #f_loss = f_loss / T.mean(T.abs(f_loss)).detach()
        #loss = loss_per_item * 1. + f_loss * self.lamda   #/ ( 1 + self.lamda)
        loss = f_loss * self.lamda

        loss = T.mean(loss)
        #loss = F.binary_cross_entropy_with_logits(logit,label,reduction="none")
        #loss = T.mean(adv_weight * loss)
        #loss.backward()
        return loss



if __name__ == "__main__":
    embed_size = 128
    feature_num = 10
    batch = 1000
    num_item = 100
    num_user = 1000
    arl = ARItem(num_user, num_item, 1.0, device="cpu")

    print(solve_m2e_equation(T.tensor(([1.0, 2.0])), T.tensor([2.0,1.5])))

    opt = T.optim.SGD(arl.parameters(),lr=1e-3)
    for i in range(batch):
        x_user = np.around(np.random.rand(10, num_item))
        x_item = np.around(np.random.rand(10, num_user))
        user_id = np.array(np.random.rand(10)*num_user, dtype=np.int32)
        item_id = np.array(np.random.rand(10)*num_item, dtype=np.int32)
        x_user = T.tensor(x_user).float()
        x_item = T.tensor(x_item).float()
        user_id = T.tensor(user_id).long()
        item_id = T.tensor(item_id).long()
        aloss = arl.adversary_step(x_item,  item_id)
        opt.step()
        loss = arl.learner_step(x_user, user_id, scale=1)
        opt.step()
        #print(x)
        print(aloss, loss)
    print(arl.item_weight, arl.item_loss)

        