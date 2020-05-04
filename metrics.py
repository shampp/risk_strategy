from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import logging

logger = logging.getLogger(__name__)

def recall(rmd,N):
    return rmd.sum()/float(N)
###########################################################################################
def precision(rmd):
    return rmd.mean()
###########################################################################################
def get_user_metrics_from_list(user_metric_dict,u_ind,k,true_m_ind,pred_m_ind,N):
    b_ratings = np.array([1 if int(i) in true_m_ind else 0 for i in pred_m_ind[:k]])
    user_metric_dict[k]['prec'][u_ind] = precision(b_ratings)
    user_metric_dict[k]['ndcg'][u_ind] = ndcg(b_ratings)
    user_metric_dict[k]['auc'][u_ind] = auc(b_ratings)
    user_metric_dict[k]['recl'][u_ind] = recall(b_ratings,N)
    return 0;
###########################################################################################
def get_user_metrics(user_metric_dict,U_rmd,u_ind,k,R_train,R_test,m_ind,N):
    rmd_ids = U_rmd[u_ind][:k]
    ratings = R_test[u_ind,rmd_ids]
    b_ratings = ratings>3
    #relev_rmd_ids,ratings = relev_rmd_index(R_test,u_ind,rmd,4)
    user_metric_dict[k]['prec'][u_ind] = precision(b_ratings)
    user_metric_dict[k]['ndcg'][u_ind] = ndcg(b_ratings)
    user_metric_dict[k]['auc'][u_ind] = auc(b_ratings)
    user_metric_dict[k]['recl'][u_ind] = recall(b_ratings,N)
    return 0;
###########################################################################################
def auc(rmd):
    try:
        res = roc_auc_score(y_true=rmd, y_score=[1]*len(rmd), max_fpr=0.5)
    except Exception:
        res = 0.0
    return res
#########################################################################################
def dcg(rmd_indx,relev_rmd_indx):
    dcg = [(math.pow(2,rmd_indx[i] in relev_rmd_indx) -1)/math.log(i+2) for i in xrange(len(rmd_indx))]
    return sum(dcg)
###########################################################################################
def ndcg(rmd):
    dcg = np.sum(rmd / np.log2(np.arange(2,len(rmd)+2)))
    sorter = sorted(rmd,reverse=True)
    m_dcg = np.sum(sorter / np.log2(np.arange(2,len(sorter)+2)))
    try:
        ndc = dcg/m_dcg
    except Exception:
        ndc = 0.0
    return ndc
###########################################################################################
