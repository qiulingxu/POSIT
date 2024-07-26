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

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return T.sparse.FloatTensor(i, v * dv, s.size())

class EASE(nn.Module):
    def __init__(self, n_dim, m_dim=0, eps=1.0, xent=False):
        super().__init__()
        if m_dim==0:
            m_dim = n_dim
        #t = T.zeros((n_dim, m_dim), layout=T.sparse_coo)
        #T.normal(0,1,size=(n_dim,m_dim), requires_grad=True))
        self.U = nn.Parameter(eps*T.normal(0,1,size=(n_dim,m_dim)), requires_grad=True)
        self.xent=xent
    
    def train_pred(self,X):
        #with T.no_grad():
        #    self.U.data.fill_diagonal_(0.)
        return self.forward(X)

    def forward(self, X, logits=True):
        with T.no_grad():
            self.U.data.fill_diagonal_(0.)
        if X.is_sparse:
            Y = T.sparse.mm(X, self.U)
        else:
            Y = T.matmul(X, self.U,)
        if self.xent:
            Y = Y -4.0
        if self.xent and not logits:
            Y = T.sigmoid(Y)
        return Y

        #return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.zeros(self.U.shape[0],1)), a_is_sparse=True)

    #def max_pred(self,X):
    #    return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.reduce_max(self.U,axis=0)), a_is_sparse=True)

if __name__=="__main__":
    m = EASE(10, eps = 1.0/np.sqrt(10))
    A = np.array(np.random.rand(10,10), dtype=np.float32)
    A = T.tensor(A)
    loss = T.mean(T.square(m(A)-A))
    print(loss)