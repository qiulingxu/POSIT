
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np

from .ease import EASE

from .arlitem_adv import ARItem_Adversary

class ARItem(nn.Module):
    def __init__(self, u_num,  i_num, lamda, loss, adv_struct, device="cpu", ):
        super().__init__()
        #self.group_weight = nn.Embedding(self.num_leaves, 1, device=self.device)
        self.u_num = u_num
        self.i_num = i_num
        self.lamda = lamda
        self.register_buffer("item_weight", T.ones((i_num,)))
        self.register_buffer("item_loss", T.zeros((i_num,)))
        self.register_buffer("loss_count", T.tensor(1e-8))
        if loss == "l2":
            self.learner = EASE(i_num, eps=0.)
        elif loss == "xent":
            self.learner = EASE(i_num, eps=0., xent=True)
        self.adversary = ARItem_Adversary(u_num, lamda=lamda, adv_struct=adv_struct)
        self.learner.to(device)
        self.adversary.to(device)
        self.device = device
        self.moment = 0.9
        self.loss = loss
        

    def forward(self, x:np.ndarray):
        o = self.learner(x)
        return o    


    def adversary_step(self, x_items, item_id):
        self.adversary.zero_grad()
        item_weight = self.adversary(x_items)

        #self.item_weight[item_id] = item_weight
        assert len(x_items) == len(item_id)
        with T.no_grad():
            self.item_weight.scatter_(dim =0, index = item_id, src=item_weight)
        #select_slice = item_weight
        
        full_item_weight = item_weight#self.item_weight#self.item_weight.scatter(dim =0, index = item_id, src=item_weight)
        full_item_weight = self.adversary.normalize(full_item_weight)
        tot_loss = full_item_weight * self.item_loss.index_select(dim=0,index=item_id)#.index_select(dim=0,index=item_id)
        
        #loss = F.binary_cross_entropy_with_logits(logit,label,reduction="none").detach()
        loss = -T.mean(tot_loss)
        #loss.backward()
        return loss

    def adversary_step_full(self, x_items):
        #self.adversary.zero_grad()
        item_weight = self.adversary(x_items)

        with T.no_grad():
            self.item_weight = item_weight.detach()
        nitem_weight = self.adversary.normalize(item_weight)
        loss = self.item_loss / self.loss_count
        tot_loss = nitem_weight * (loss.detach())#
        loss = -T.mean(tot_loss) #*1e3
        return loss

    def learner_step(self, X, user_id, scale):
        pred = self.learner(X)
        if self.loss == "l2":
            loss = T.square(pred - X) 
        elif self.loss == "xent":
            Xd = X.to_dense()
            loss = F.binary_cross_entropy_with_logits(pred,Xd,reduction="none")
        else:
            assert False
        loss_per_item = T.mean(loss, dim=0) * scale
        #
        with T.no_grad():
            self.item_loss = self.moment*self.item_loss  + loss_per_item#
            self.loss_count = self.loss_count * self.moment + 1.0

        item_weight = self.adversary.normalize(self.item_weight)
        loss = T.mean(loss_per_item * (1.0+ item_weight * self.lamda))
        return loss


