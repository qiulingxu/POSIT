"""import tensorflow as tf
class Model(object):
    def __init__(self, n_dim, m_dim=0, eps=0.0):
        if m_dim==0:
            m_dim = n_dim
        self.U = tf.Variable(eps*tf.random.normal(shape=[n_dim,m_dim]))
    def pred(self,X):
        return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.zeros(self.U.shape[0],1)), a_is_sparse=True)
    def max_pred(self,X):
        return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.reduce_max(self.U,axis=0)), a_is_sparse=True)
        
class TrainParams(object):
    def __init__(self, C, B, neg_weight=1.0, learning_rate=0.001, num_iter=10000, batch_size=1024, check_progress_iter=1000, min_iter=5000):
        self.C = tf.constant(C)
        self.B = tf.constant(B)
        self.neg_weight = tf.constant(neg_weight)
        self.learning_rate = tf.constant(learning_rate)
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.check_progress_iter = check_progress_iter
        self.min_iter = min_iter"""

import torch as T
import torch.nn as nn
import numpy as np

class Tail(nn.Module):
    def __init__(self, learner, loss, beta=0.1, lamda=1.0, leakiness= 0.01, alphascale=1e3):
        super().__init__()
        self.learner = learner
        self.loss = loss
        self.alpha = nn.Parameter(T.zeros(size=(1,)), requires_grad=True)
        self.lamda = lamda
        self.beta = beta
        self.alphascale = alphascale
        # I use the leaky relu to make sure there will be gradient for negative side as well
        self.pos_op = T.nn.ReLU()#T.nn.LeakyReLU(negative_slope=leakiness)#

    def forward(self, X, Y=None):
        pred = self.learner(X)
        X = X.to_dense()
        if self.loss == "l2":
            loss = T.square(pred - X) 
        #loss = self.loss(X,Y)
        alpha = self.alpha / self.alphascale
        tail_opt = alpha + 1.0/self.beta * self.pos_op(loss - alpha)
        tot_loss = self.lamda * T.mean(tail_opt)
        return tot_loss
        #return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.zeros(self.U.shape[0],1)), a_is_sparse=True)

    #def max_pred(self,X):
    #    return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.reduce_max(self.U,axis=0)), a_is_sparse=True)

if __name__=="__main__":
    m = EASE(10, eps = 1.0/np.sqrt(10))
    A = np.array(np.random.rand(10,10), dtype=np.float32)
    A = T.tensor(A)
    loss = T.mean(T.square(m(A)-A))
    print(loss)