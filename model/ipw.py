
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np

try:
    from .ease import EASE
except:
    from ease import EASE

from  sklearn.linear_model import SGDRegressor

# This version we will put linear correction into the factor

class IPWItem(nn.Module):
    def __init__(self, u_num, lamda, i_num,  loss,  alpha, device="cpu", moment=0.9, **karg):
        super().__init__()
        #self.group_weight = nn.Embedding(self.num_leaves, 1, device=self.device)
        self.u_num = u_num
        self.i_num = i_num
        self.alpha = alpha
        self.loss = loss
        self.lamda = lamda


        if loss == "l2":
            self.learner = EASE(i_num, eps=0.)
        elif loss == "xent":
            self.learner = EASE(i_num, eps=0., xent=True)
        self.learner.to(device)
        self.alpha = alpha
        self.device = device
        self.moment = moment

    def forward(self, x:np.ndarray):
        o = self.learner(x)

        return o    

    def check_count(self, x_items):
        if x_items.is_sparse:
            self.counts = (T.sparse.sum(x_items, dim=1).to_dense())
        else:
            self.counts = (T.sum(x_items, dim=1))
        #self.counts = self.counts / T.min(self.counts)
        self.alpha = T.exp(-self.alpha*self.counts)
        self.item_weight = self.alpha / T.mean(self.alpha)

    def learner_step(self, x_users, user_id, scale):
        self.learner.zero_grad()
        
        pred = self.learner(x_users)
        x_users = x_users.to_dense()
        if self.loss == "l2":
            loss = T.square(pred - x_users) 
        elif self.loss == "xent":
            Xd = x_users
            loss = F.binary_cross_entropy_with_logits(pred,Xd,reduction="none")
        else:
            assert False
        loss_per_item = T.mean(loss, dim=0) * scale
 
        
        #print(loss_per_item.shape, weight1.shape)
        f_loss = loss_per_item * self.item_weight * self.lamda
        #f_loss = f_loss / T.mean(T.abs(f_loss)).detach()
        #loss = loss_per_item * 1. + f_loss * self.lamda   #/ ( 1 + self.lamda)
        loss = f_loss

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

        