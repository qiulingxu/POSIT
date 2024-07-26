import xgboost as xgb 
import torch as T
from .xgbt_nn_hybrid import DecisionTree_NN

# baseline_nn
# baseline_nn_nce
# arl_extend_1.00e+00

def get_model_for_netflix_pvr(name):
    model_name = "xgb_model_baseline_7_0522"
    if name == "baseline":
        bst = xgb.Booster()
        
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file)
        model = DecisionTree_NN(bst,eta=0.03, pretrained=True)
    elif name.find("baseline_nn")>=0 :
        bst = xgb.Booster()
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file,)
        model = DecisionTree_NN(bst,eta=0.03)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
    elif name.find("arl_extend")>=0 :
        from model.arl import ARL_Adversary, ARL
        bst = xgb.Booster()
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file,)
        learner = DecisionTree_NN(bst,eta=0.03)
        adversary = ARL_Adversary(embed_size = learner.num_leaves)
        model = ARL(learner = learner, adversary =adversary)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    elif name.find("arl4")>=0 :
        from model.arl4 import ARL_Adversary, ARL
        bst = xgb.Booster()
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file,)
        learner = DecisionTree_NN(bst,eta=0.03)
        adversary = ARL_Adversary(embed_size = learner.num_leaves)
        assert False
        model = ARL(learner = learner, adversary =adversary)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    elif name.find("agroup_extend")>=0:
        from model.agroup import AGroup, AGroup_Adversary
        bst = xgb.Booster()
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file,)
        learner = DecisionTree_NN(bst,eta=0.03)
        adversary = AGroup_Adversary(embed_size = learner.num_leaves)
        model = AGroup(learner = learner, adversary =adversary)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner        
    elif name.find("agroup2_extend")>=0:
        from model.agroup2 import AGroup, AGroup_Adversary
        bst = xgb.Booster()
        saved_model_file = 'checkpoint/{}.model'.format(model_name)
        bst.load_model(saved_model_file,)
        learner = DecisionTree_NN(bst,eta=0.03)
        adversary = AGroup_Adversary(embed_size = learner.num_leaves)
        model = AGroup(learner = learner, adversary =adversary)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner     
    else:
        assert False
    #print(state_dict["model_state_dict"].keys())
    return model


def get_model_for_common(name):
    if name.find("xent") >=0 :
        loss = "xent"
    else:
        loss = "l2"
    if name.find('ml20m')>=0:
        num_movies = 20108
        num_users = 116677
    elif name.find('nflx')>=0:
        num_movies= 17769
        num_users= 383435
    elif name.find("msd")>=0:
        num_movies = 41140
        num_users = 471355
    else:
        assert False
    if name.find("baseline_xent")>=0:
        from model.ease import EASE
        model = EASE(num_movies,xent=True)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
    elif name.find("baseline")>=0:
        from model.ease import EASE
        model = EASE(num_movies)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
    elif name.find("tail_ld")>=0 :
        from model.ease import EASE
        from model.tail_opt_ld import Tail_LD as Tail
        model = EASE(num_movies)
        tail = Tail(learner = model, i_num = num_movies, loss = loss)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        tail.load_state_dict(state_dict["model_state_dict"])
    elif name.find("tail")>=0 :
        from model.ease import EASE
        from model.tail_opt import Tail
        model = EASE(num_movies)
        tail = Tail(learner = model,loss = loss)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        tail.load_state_dict(state_dict["model_state_dict"])
    elif name.find("itemarl_ld")>=0 :
        from model.aritem_ld_common import ARItem
        model = ARItem(u_num = num_users , i_num = num_movies, lamda=1.0, moment=0.95, loss=loss)

        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    elif name.find("itemarl_metric")>=0 :
        from model.aritem_recall1s_common import ARItem
        pos = name.find("_ast")
        if  pos == -1:
            advstruct = "linearbn"
        else:
            advstruct = name[pos + 4:]
            advstruct = advstruct[:advstruct.find("_")]
        nstd = name[name.find("_nstd")+5:]
        nstd = nstd[:nstd.find("_")]
        nstd = float(nstd)
        
        print(advstruct)
        model = ARItem(u_num = num_users , i_num = num_movies, lamda=1.0, moment=0.95, nstd=nstd, adv_struct=advstruct, ones_aware=True, adv_loss="l2", loss=loss)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        #print(state_dict["model_state_dict"])
        #print(model.state_dict())
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    elif name.find("itemarl")>=0 :
        from model.aritem_common import ARItem
        model = ARItem(u_num = num_users , i_num = num_movies, lamda=1.0,  adv_struct="linearbn", loss=loss)
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    elif name.find("ipw")>=0 :
        from model.ipw import IPWItem
        model = IPWItem(u_num = num_users , i_num = num_movies, lamda=1.0, alpha=1.0, loss="l2")
        state_dict = T.load("checkpoint/{}.pt".format(name), map_location=T.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.learner
    else:
        assert False
    #print(state_dict["model_state_dict"].keys())
    return model
