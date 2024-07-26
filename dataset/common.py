from re import L
import os
import json
import functools
import regex as re
import scipy.sparse

import torch as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.data import buffer_iter


def load_movie_lens_data(path='', file_name='', num_movies=20108, num_users=136677, user_id={}, movie_id={}, logging=False):
   allx = []
   allusers = []
   allitems = []
   if len(user_id) > 0:
       allx = [[0.0]*num_movies for _ in range(0,len(user_id))]
   with open('ml-20m-refined/%s' %(file_name), 'r') as file:
      lines1 = file.read().split('\n')
      k = 0
      for line in lines1:
         line = line.replace('\r', '').split(',')
         if len(line)<2:
            break
         k = k + 1
         if k==1:
            continue
         # Use str, otherwise json will transform it to str and reloading will fail
         movie = str(int(line[1]))
         user = str(int(line[0]))
         #print( movie in movie_id, movie, movie_id)
         #exit()
         if user in user_id:
             this_user_id = user_id[user]
         else:
             this_user_id = len(user_id)
             user_id[user] = len(user_id)
             if (this_user_id < num_users):
                allx.append([0.0]*num_movies)
                allusers.append(user)
         if movie in movie_id:
             this_movie_id = movie_id[movie]
         else:
             this_movie_id = len(movie_id)
             movie_id[movie] = len(movie_id)
             if (this_movie_id < num_movies):
                allitems.append(movie)
         #print(movie, user, num_movies, num_users, len(movie_id), this_movie_id, this_user_id,)
         if (this_movie_id < num_movies) and (this_user_id < num_users):
             if logging:
                 print(this_user_id,this_movie_id)
             allx[this_user_id][this_movie_id] = 1.0
             #print("in")
         #exit()
            

   print("number of lines:", k, "number of users:", len(user_id), "number of movies:", len(movie_id))
   return (allx, user_id, movie_id, allusers, allitems)


def get_cache_npy(f, file_path):
    # cache the data for fast retrival # No cache needed for now, it is fast enough to load these datasets
    if os.path.exists(file_path + "_dct.json"):
        ret = np.load(file_path + "_lst.npz")
        sparse_matrix = scipy.sparse.load_npz(file_path + "_mat.npz")
        #mat1 = sparse_matrix#.toarray()
        l1 = ret["l1"]
        l2 = ret["l2"]
        js = json.loads(open(file_path + "_dct.json","r").read())
        d1 = js["d1"]
        d2 = js ["d2"]
    else:
        mat1, d1, d2, l1, l2 = f()
        mat1 = np.array(mat1, dtype=np.float32)
        sparse_matrix = scipy.sparse.csc_matrix(mat1)
        scipy.sparse.save_npz(file_path + "_mat",sparse_matrix)
        np.savez(file_path + "_lst", l1=l1,l2=l2)
        open(file_path + "_dct.json", "w").write(json.dumps({"d1":d1, "d2":d2}))
    return sparse_matrix, d1, d2, l1, l2

def get_all_data_py(dataset='ml20m'):

    if dataset=='ml20m':
        num_movies=20108
        num_users=136677
        path = 'NA'
    elif dataset=='nflx': 
        num_movies=17769
        num_users=463435 
        path = 'NA'
    elif dataset=='msd':
        num_movies=41140
        num_users=571355         
        path = 'NA'
    else:
        assert False
    
    os.makedirs("./data/common",exist_ok=True)
    
    (train, user_id, movie_id, train_users, _) = get_cache_npy(functools.partial(load_movie_lens_data, path=path, file_name='train.csv',num_movies=num_movies, num_users=num_users), "./data/common/{}_train".format(dataset))
    #print(train.shape,len(movie_id) )
    #train = None
    #train = tf.constant(train)
    #del train
    (test_tr, user_id, movie_id, test_tr_users, _) =get_cache_npy(functools.partial(load_movie_lens_data,file_name='test_tr.csv',path=path,num_movies=num_movies, num_users=num_users, user_id={}, movie_id=movie_id, logging=False),  "./data/common/{}_test_tr".format(dataset))
    #test_tr = np.array(test_tr)
    #test_tr = tf.constant(test_tr)
    

    (test_te, user_id, movie_id, test_te_users, _) =get_cache_npy(functools.partial(load_movie_lens_data,file_name='test_te.csv',path=path,num_movies=num_movies, num_users=num_users, user_id=user_id, movie_id=movie_id, logging=False), "./data/common/{}_test_te".format(dataset))
    test_te = test_te.toarray()
    test_tr = test_tr.toarray()
    test_te = test_te - 100*(np.greater(test_tr, 0.))
    #T.tensor(test_te) - 100*T.cast(test_tr > 0, dtype=tf.float32)
    #test_tr = tf.constant(test_tr)
    print(test_te.shape,len(movie_id) )
    #test_te = tf.constant(test_te) - 100*tf.cast(test_tr > 0, dtype=tf.float32)

    (valid_tr, user_id, movie_id, valid_tr_users, _) =get_cache_npy(functools.partial(load_movie_lens_data,file_name='validation_tr.csv',path=path, num_movies=num_movies, num_users=num_users, user_id={}, movie_id=movie_id),  "./data/common/{}_valid_tr".format(dataset))
    #valid_tr = tf.constant(valid_tr)
    #np.array(valid_tr)
    
    (valid_te, user_id, movie_id, valid_te_users, _) =get_cache_npy(functools.partial(load_movie_lens_data,file_name='validation_te.csv',path=path, num_movies=num_movies, num_users=num_users, user_id=user_id, movie_id=movie_id), "./data/common/{}_valid_te".format(dataset))
    valid_te = valid_te.toarray()#np.array(valid_te)
    valid_tr = valid_tr.toarray()
    valid_te = valid_te - 100 * ( np.greater(valid_tr, 0.) )
    #valid_te = tf.constant(valid_te) - 100*tf.cast(valid_tr > 0, dtype=tf.float32)
    print(valid_te.shape, len(movie_id))

    return (train, test_tr, test_te, valid_tr, valid_te, movie_id)

import csv
def get_movie_lens_tags(path="NA", movie_ids={}):

    num_movies = 20108
    with open("./data/ml-20m/movies.csv") as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)[1:]
        str1=open("unique_sid.txt", "r").read() 
        unique_sid = str1.split("\n")    
    
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    p = re.compile(r"\((\d+)\)")
    genres = {}
    #print(movie_ids)
    tags = {}
    for line in lines:

        movie_id, movie_name, genre_lst = line
        if movie_id not in show2id:
            continue
        movie_id = str(show2id[movie_id])
        #if movie_id not in movie_ids:
        #    continue
        #movie_id = str(movie_ids[movie_id])
        #movie_id = int(movie_id)

        year = p.findall(movie_name)
        if not year:
            print(movie_name, "doesn't have year")
            # I randomly assign a year because only a few is this case
            year= [2009]
        year = int(year[-1])
        ## Assume only the first category counts for simpilcity
        genre_lst = genre_lst.split("|")
        genre_lst = [genre.strip() for genre in genre_lst]
        for genre in genre_lst:
            if genre not in genres:
                genres[genre] = len(genres) 
        genre_lst = [genres[genre] for genre in genre_lst]
        tags[movie_id] = [year,genre_lst]
        #print(movie_id, year, genres[genre])
    for i in range(num_movies):
        assert str(i) in tags, "data {} has no tags matched.".format(unique_sid[i])
    
    open("data/common/ml20m_movie_tags.json","w").write(json.dumps({"tag":tags, "genres":genres}))
    print(genres)
    ytag = np.zeros([num_movies,], dtype=np.int16)
    for i in range(num_movies):
        ytag[i] = tags[str(i)][0]
    np.save("data/common/ml20m_year_tag", ytag)
    # Handle the multipe label, -1 means the slot is vacant
    ytag = np.zeros([num_movies,10], dtype=np.int16) -1
    for i in range(num_movies):
        for idx,genre in enumerate(tags[str(i)][1]):
            ytag[i,idx] = genre
    np.save( "data/common/ml20m_genre_tag", ytag)    

def get_nflx_tags(path="NA", movie_ids={}):
    #with S3(s3root=path) as s3:
    #    str1=s3.get('%s' %("movies.dat"))  
    num_movies = 17769
    with open("./data/nflx/movie_titles.csv", encoding="ISO-8859-1") as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)#[1:]

    with S3(s3root="NA") as s3:
        str1=s3.get('%s' %("unique_sid.txt")) 
        unique_sid = str1.text.split("\n")    
    
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    p = re.compile(r"\((\d+)\)")
    genres = {}
    #print(movie_ids)
    tags = {}
    for line in lines:
        line = line[:3]
        movie_id, year, movie_name = line
        if movie_id not in show2id:
            continue
        movie_id = str(show2id[movie_id])
        #if movie_id not in movie_ids:
        #    continue
        #movie_id = str(movie_ids[movie_id])
        #movie_id = int(movie_id)

        """year = p.findall(movie_name)
        if not year:
            print(movie_name, "doesn't have year")
            # I randomly assign a year because only a few is this case
            year= [2009]"""
        try:
            year = int(year)
        except: 
            year = 2009
        ## Assume only the first category counts for simpilcity
        tags[movie_id] = year
        #print(movie_id, year, genres[genre])
    
    open("data/common/nflx_movie_tags.json","w").write(json.dumps({"tag":tags}))

    ytag = np.zeros([num_movies,], dtype=np.int16)
    for i in range(num_movies):
        ytag[i] = tags[str(i)]
    np.save("data/common/nflx_year_tag", ytag)

def covert_csr_to_torch_coo(csr_arr):
    from scipy.sparse import coo_matrix

    coo = coo_matrix(csr_arr)

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = T.LongTensor(indices)
    v = T.FloatTensor(values)
    shape = coo.shape

    sparse = T.sparse.FloatTensor(i, v, T.Size(shape))
    return sparse

def covert_csr_to_torch_csr(csr_arr):
    """from scipy.sparse import coo_matrix

    #coo = coo_matrix(csr_arr)

    values = csr_arr.data

    sparse = T.sparse_csr_tensor(csr_arr.indptr, csr_arr.indices, values)
    return sparse"""
    from scipy.sparse import coo_matrix

    coo = coo_matrix(csr_arr)

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = T.LongTensor(indices)
    v = T.FloatTensor(values)
    shape = coo.shape

    sparse = T.sparse.FloatTensor(i, v, T.Size(shape)).to_sparse_csr()
    return sparse        

if not T.cuda.is_available():
    covert_csr_to_torch = covert_csr_to_torch_csr
else:
    covert_csr_to_torch = covert_csr_to_torch_coo

def shuffle_csr(matrix, index=None):
    rindex = np.arange(np.shape(matrix)[0])
    np.random.shuffle(rindex)
    if index is not None:
        return matrix[rindex, :], index[rindex]
    else:
        return matrix[rindex, :]

def shuffle_csr_column(matrix, index=None):
    rindex = np.arange(np.shape(matrix)[1])
    np.random.shuffle(rindex)
    if index is not None:
        return matrix[:, rindex], index[rindex]
    else:
        return matrix[:, rindex]


class MovieLenDataset(Dataset):
    def  __init__(self, dataset = "ml20m", batch_size=64, shuffle=False, tags=False):
        train, self.test_tr, self.test_te, self.valid_tr, self.valid_te, self.movie_id = get_all_data_py(dataset)
        if dataset=='ml20m':
            num_movies=20108
            num_users=136677
            path = 'NA'
            tag_path = "./data/ml_len_mtag.json"
            num_num_feat = 1
            num_category_feat = 1
        elif dataset=='nflx': 
            num_movies=17769
            num_users=463435 
            path = 'NA'
        elif dataset=='msd':
            num_movies=41140
            num_users=571355         
            path = 'NA'
        else:
            assert False

        assert shuffle == True
        #self.return_index = return_index
        self.user_index =  np.arange(num_users, dtype=np.int32)
        self.item_index = np.arange(num_movies, dtype=np.int32)
        self.tags = tags
        # minuse 20000 for training data
        if dataset == "msd":
            self.max_user = num_users - 100000
        elif dataset == "nflx":
            self.max_user = num_users - 80000
        else:
            self.max_user = num_users - 20000
        self.max_item = num_movies
        self.shuffle =shuffle
        if train is not None:
            self.train_user =train.copy()
            self.train_original =train.copy()
            self.train_item = train.copy()
        #print(self.train_item.shape, train.shape)
        self.batch_size = batch_size
        self.batch_num_user = self.train_user.shape[0] // self.batch_size
        self.batch_num_item = self.train_item.shape[1] // self.batch_size
        if self.tags:
            js = json.loads(open(tag_path, "r").read())
            js = js["tag"]
            self.num_tag = np.zeros(shape=(num_movies,num_num_feat))
            self.cate_tag = np.zeros(shape=(num_movies,num_category_feat))
            for mid, v in js.items():
                #print(v)
                mindex = int(mid)#self.movie_id[str(mid)]
                self.num_tag[mindex,0] = int(v[0])
                self.cate_tag[mindex,0] = v[1]
    def __len__(self):
        return self.batch_num_user

    def __getitem__(self, index):
        return covert_csr_to_torch(self.train_user[index:index+1])

    def __iter__(self):
        return self.iter_on_user()#buffer_iter(8, self.iter_on_user,lambda x:x)#

    def iter_on_user(self):
        if self.shuffle:
            self.train_user, self.user_index = shuffle_csr(self.train_user, index=self.user_index)

        for i in range(self.batch_num_user):
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            yield covert_csr_to_torch(self.train_user[s:e]), self.user_index[s:e]

    def iter_on_item(self):
        if self.shuffle:
            self.train_item, self.item_index = shuffle_csr_column(self.train_item, index=self.item_index)

        for i in range(self.batch_num_item):
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            yield covert_csr_to_torch(self.train_item[:,s:e].transpose()), self.item_index[s:e]

def forever_iter(f):
    while True:
        for i in f():
            yield i

if __name__ == "__main__":
    #Code for genrating tags
    get_nflx_tags()
    exit()
    train, test_tr, test_te, valid_tr, valid_te, movie_id = get_all_data_py("ml20m")
    print(train[0])
    #get_movie_lens_tags(movie_ids=movie_id)
    #for data in MovieLenDataset("ml20m", batch_size=100):
    #    print(data)
        
    ml = MovieLenDataset("ml20m", tags=True, shuffle=True)
    for i, ind in ml.iter_on_item():
        print(ind)