#!/usr/bin/env python3
import sys
import os.path as path
import numpy as np
from time import time
import  scipy.spatial.distance as distance
import logging
from metrics import get_user_metrics
from nmf import als_nmf
from algo import *
from joblib import Parallel,delayed,cpu_count
from data_utils import *

np.seterr(invalid='raise')
baselines = {
    'sca' : sca,   #Extended Greedy algorithm with collaborative filtering
}

sca_func = 'risks' # can be powe pow1 pow9 sqrt log riska1 riska5 riska01 riskae risks
###########################################################################################
    
def main(alg): #main function
    max_rat = 5 #maximum utility value
    R = U = V = Rhat = None
    tot_itr = 5
    rmd_sz = [3,5,10,20]  #metrics will be calculated for this recommendation.
    K = max(rmd_sz)
    all_datasets = ['dataset'+str(_) for _ in range(4,5)]
    random = np.random.RandomState(seed=tot_itr)
    for dataset in all_datasets:
        result_dict = dict()
        for i in rmd_sz:
            result_dict[i] = dict()
            result_dict[i]['prec'] = list() #precision
            result_dict[i]['ndcg'] = list() #ndcg
            result_dict[i]['auc'] = list()  #auc
            result_dict[i]['recl'] = list()

        folder_path = get_folder_path(dataset)
        dr = get_ratings_df(dataset)    #this contains ratings table containing users with more than 100 relevant ratings
        #dm = get_movies_df(dataset)
        #dm = filter_movies(dr,dm)   # we need to filter out the movies df to contain used with more than 100 (positive ratings only)
        tr_df,te_df = split_train_test(dr)
        te_df = te_df[te_df.IID.isin(tr_df.IID)]    #we have to remove the test items which is not part of training set
        tr_df = tr_df[tr_df.IID.isin(te_df.IID)]
        print("Number of relevant ratings in training set")
        print(tr_df[tr_df.R == 1].count())
        print("Number of relevant ratings in test set")
        print(te_df[te_df.R == 1].count())
        print("Size of the smallest user group in training set: %d" %tr_df[tr_df.R == 1].groupby(['UID']).size().min())
        print("Size of the largest group in training set: %d" %tr_df[tr_df.R == 1].groupby(['UID']).size().max())
        print("Size of the smallest group in test set: %d" %te_df[te_df.R ==1].groupby(['UID']).size().min())
        print("Size of the largest group in test set: %d" %te_df[te_df.R == 1].groupby(['UID']).size().max())
        write_to_file(tr_df,'train.txt')
        write_to_file(te_df,'test.txt')
        R_train = get_rating_matrix(tr_df)
        R_test = get_rating_matrix(te_df)
        print("Shape of the Matrix")
        print(R_train.shape)
        assert R_train.shape == R_test.shape
        curr = time()
        (r_sz,c_sz) = R_train.shape
        if alg == 'sca':
            log_file = folder_path + '%s_%s_access.log' %(alg,sca_func)
        else:
            log_file = folder_path + '%s_access.log' %(alg)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        U_rmd = dict()
        user_metric_dict = dict()
        for i in rmd_sz:
            user_metric_dict[i] = dict()
            user_metric_dict[i]['prec'] = dict()
            user_metric_dict[i]['ndcg'] = dict()
            user_metric_dict[i]['auc'] = dict()
            user_metric_dict[i]['recl'] = dict()

        users = get_test_users(R_test)  #get only users who has test ratings.
        U,V = als_nmf(R_train,folder_path)
        II_SiM = get_item_item_sim(V.T)  #V is of dimension (k,n)

        for u_ind in users:
            m_ind,m_rat = get_rated_movies(R_train,u_ind)   #get already rated movie indxs and ratings values
            Unrated_M,N = get_unrated_movies(R_test,u_ind)
            U_rmd[u_ind] = baselines[alg](Unrated_M,II_SiM,m_ind,m_rat,K)
 
            for i in rmd_sz:
                logging.info("Getting user metrics for user id: %d and recommendation size: %d" %(u_ind,i))
                get_user_metrics(user_metric_dict,U_rmd,u_ind,i,R_train,R_test,m_ind,N)
        logging.info("============================================================================")
        del R_train
        del R_test
        for i in rmd_sz:
            logging.info("Top %d Recommendations" %(i))
            result_dict[i]['prec'].append(np.mean(list(user_metric_dict[i]['prec'].values())))
            logging.info("Average Precision: %f" %(result_dict[i]['prec'][-1]))
            result_dict[i]['ndcg'].append(np.mean(list(user_metric_dict[i]['ndcg'].values())))
            logging.info("Average NDCG : %f" %(result_dict[i]['ndcg'][-1]))
            result_dict[i]['auc'].append(np.mean(list(user_metric_dict[i]['auc'].values())))
            logging.info("Average AUC : %f" %(result_dict[i]['auc'][-1]))
            result_dict[i]['recl'].append(np.mean(list(user_metric_dict[i]['recl'].values())))
            logging.info("Average Recall : %f" %(result_dict[i]['recl'][-1]))
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)                              
        #here put the code to do average over the total iterations
    for i in rmd_sz:
        print("============================================================================")
        print("============================================================================")
        print("%d recommendations for %s algorithm (%d iterations)" %(i,alg,tot_itr))
        print("Averag Precision: %f" %(np.mean(result_dict[i]['prec'])))
        print("Averag NDCG: %f" %(np.mean(result_dict[i]['ndcg'])))
        print("Averag AUC: %f" %(np.mean(result_dict[i]['auc'])))
        print("Averag Recall: %f" %(np.mean(result_dict[i]['recl'])))
    
if __name__ == '__main__':
    for alg in baselines.keys():
        main(alg)
