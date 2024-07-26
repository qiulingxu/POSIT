# rec-promote-semantic-tail

## Introduction
This is the implementation of Paper "Promotion of Semantic Item Tail via Adversarial Learning"


In this project, we explored the ways of generating an inter-user diversified recommendations using an adversarial approach. In many recommender problems, a handful of popular items (e.g. movies/TV shows, news etc.) can be dominant in recommendations for many users. However, we know that in a large catalog of items, users are likely interested in more than what is popular. The dominance of popular items may mean that users will not see items they would likely enjoy. In this paper, we propose a technique to overcome this problem using adversarial machine learning. We define a metric to translate user-level utility metric in terms of an advantage/disadvantage over items. We subsequently use that metric in an adversarial learning framework to systematically promote disadvantaged items.  The resulting algorithm identifies semantically meaningful items that get promoted in the learning algorithm. In the empirical study, we evaluate the proposed technique on three publicly available datasets and four competitive baselines. The result shows that our proposed method not only improves the coverage, but also, surprisingly, improves the overall performance.

Our method can greatly improve the coverage on the public datasets. As an example, here shows the result from Netflix Prize Dataset.

![Result on Netflix Prize Dataset](netflix_prize.png)

## How to Use

In the common datasets, we provide a one-stop shop for all the methods including Ours, IPW, EASE and Tail Optimization and all datasets including Movielens20, Netflix Prize and MSD. The main body is written in train_body.py. There are two interfaces to use the code, one is through metaflow and the other is through notebooks.

To use it on notebooks, use the file train_common.ipynb. It will call train_body from notebooks.

The parameters required to reproduce our results.

|Dataset|Params|
|-----|-----|
|MovieLens |	ml20m_ease_lr2.00e+00_alr1.00e+00_wd8.00e-06_nstd1.50e+00_awd0.00e+00_ast2bmlp2_alossrecall_dist_onesFalse_lam1.00e+00_l2_itemarl_metric |
|Netflix Prize |nflx_ease_lr1.00e+00_alr1.00e+00_wd5.00e-05_nstd1.00e+00_awd0.00e+00_ast2mlp50_alossrecall_dist_onesFalse_lam1.00e+00_l2_itemarl_metric |
|Million Song | msd_ease_lr4.00e+01_alr1.00e-01_wd1.00e-05_nstd5.00e-01_awd1.00e-05_ast2mlp2_alossrecall_onesTrue_lam1.00e+00_l2_itemarl_metric|


You can fill these parameters in train_body.py or jupyter notebook and invoke train_body function to train.

## Dependency

The program will auto run on GPU if the device is avaiable and on CPU otherwise.
It requires pytorch. We modified the python package recsys_metrics https://github.com/zuoxingdong/recsys_metrics for evaluation.