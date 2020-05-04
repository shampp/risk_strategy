import numpy as np;
from numpy import sqrt,square,log;
from time import time;
from scipy.linalg import norm;
import logging;

logger = logging.getLogger(__name__)

def dum(Unrated_M,SiM,m_ind,mind_R,K,u_ind,Rhat,MG_Mat):
    logger.info('DUM Algorithm')
    Rec_Movies_ID = list();
    indx = np.argsort(Rhat[u_ind,Unrated_M])[::-1];
    gen_sum = np.sum(MG_Mat[Unrated_M[indx],:],axis=1); #get genre count in the order of decreasing relevance
    ind = 0;    #most relevant item is added to the list first.
    val = 0;
    for i in range(K):
        best_select = Unrated_M[indx[ind]];
        Rec_Movies_ID.append(best_select);
        val = gen_sum[ind];
        gen_sum -= val;
        gen_sum = np.delete(gen_sum,ind);
        indx = np.delete(indx,ind);
        ind = np.argmax(gen_sum);
    logger.info('Finished DUM re-ranking algorithm')
    return np.array(Rec_Movies_ID,dtype=int),0;
###########################################################################################
def msd(Unrated_M,SiM,m_ind,mind_R,K,u_ind,Rhat,MG_Mat):
    logger.info('MSD Algorithm')
    eps = 1e-8
    Rec_Movies_ID = list()
    CG = list()
    beta = 0.5
    S = np.array([],dtype=np.uint16)
    xf = Rhat[u_ind,Unrated_M]+beta*np.sum(1-SiM[:,Unrated_M],axis=0) + eps
    F_j = np.copy(xf)
    xf_b = 0
    for i in range(K):
        A = xf-xf_b
        CG.append(1-np.min(A/F_j))
        ind = np.argmax(A)
        best_select = Unrated_M[ind]
        S = np.append(S,best_select)
        Rec_Movies_ID.append(best_select)
        Unrated_M  = np.delete(Unrated_M,ind)
        F_j = np.delete(F_j,ind)
        xf_b = np.delete(xf,ind)
        xf = Rhat[u_ind,Unrated_M]+beta*np.sum(1-SiM[S[:,None],Unrated_M],axis=0)
    logger.info('Finished MSD re-ranking algorithm')
    return np.array(Rec_Movies_ID,dtype=int),max(CG)
###############################################################################################
def mmr(Unrated_M,SiM,m_ind,mind_R,K,u_ind,Rhat,MG_Mat):
    logger.info('MMR algorithm')
    eps = 1e-8
    mm = 8
    beta = 0.5
    Rec_Movies_ID = list() #copy of the already rated movie indexes
    CG = list()
    S = np.array([],dtype=np.uint16)
    xf = beta*(mm+Rhat[u_ind,Unrated_M])+ eps#-(1-beta)*np.max(SiM[:,Unrated_M],axis=0) + eps
    F_j = np.copy(xf)
    beta = 0.5
    for i in range(K):
        CG.append(1-np.min(xf/F_j))
        ind = np.argmax(xf)
        best_select = Unrated_M[ind]
        S = np.append(S,best_select)
        Rec_Movies_ID.append(best_select)
        Unrated_M  = np.delete(Unrated_M,ind)
        F_j = np.delete(F_j,ind)
        xf = beta*(mm+Rhat[u_ind,Unrated_M])-(1-beta)*np.max(SiM[S[:,None],Unrated_M],axis=0) + eps
    logger.info('Finished MMR re-ranking algorithm')
    return np.array(Rec_Movies_ID,dtype=int),max(CG)
#################################################################################################
def sca(Unrated_M,SiM,m_ind,mind_R,K): #greedy and b_greedy are in fact same.
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    sca_func = {
                'pow1'   : np.power,
                'pow9'  : np.power,
                'sqrt'  : np.sqrt,
                'log'   : np.log,
                'riska'  : np.exp,  #seek aversion
                'risks'  : np.exp,  #seek seeking
            }
    c_f = 'risks'
    #logger.info('SCA algorithm with concave function: %s' %(c_f))
    eps = 1e-8
    Rec_Movies_ID = list() #Greedy Set
    sum_W = np.zeros((Unrated_M.shape[0],m_ind.shape[0]))
    Tsum_W = sum_W + SiM[Unrated_M[:,None],m_ind]
    if c_f == 'powe':
        xf = np.dot(np.power(Tsum_W,0.01),mind_R) +eps#xf contains fucntion values for all the unrateed items
    if c_f == 'pow1':
        xf = np.dot(np.power(Tsum_W,0.1),mind_R) +eps#xf contains fucntion values for all the unrateed items
    if c_f == 'pow9':
        xf = np.dot(np.power(Tsum_W,0.9),mind_R) +eps#xf contains fucntion values for all the unrateed items
    if c_f == 'sqrt':
        xf = np.dot(np.sqrt(Tsum_W),mind_R) +eps#xf contains fucntion values for all the unrateed items
    if c_f == 'log':
        xf = np.dot(np.log(1+Tsum_W),mind_R) + eps
    #for risk aversion we try with 1, 0.1, 0.01
    if c_f == 'riska1':
        xf = np.dot(1-np.exp(-1*Tsum_W),mind_R) + eps
    if c_f == 'riska5':
        xf = np.dot(1-np.exp(-0.5*Tsum_W),mind_R) + eps
    if c_f == 'riska01':
        xf = np.dot(1-np.exp(-0.1*Tsum_W),mind_R) + eps
    if c_f == 'riskae':
        xf = np.dot(1-np.exp(-0.001*Tsum_W),mind_R) + eps
    if c_f == 'risks':
        xf = -(np.dot(1-np.exp(1*Tsum_W),mind_R) + eps)
    F_j = np.copy(xf)  #F_i for all the items in the unrated set
    xf_b = 0
    for i in range(K):  #iterate over the set of unrated movies.
        ind = np.argmax(xf)
        sum_W = Tsum_W[ind,:]
        best_select = Unrated_M[ind]
        Rec_Movies_ID.append(best_select) #We add the best selected to the array.
        Unrated_M  = np.delete(Unrated_M,ind)   #this is ok, since indexes are unique
        F_j = np.delete(F_j,ind)  #remove the item selected by the greedy strategy
        xf_b = np.delete(xf,ind)
        Tsum_W = sum_W + SiM[Unrated_M[:,None],m_ind]
        #xf = np.dot(sqrt(Tsum_W),mind_R)+eps #xf contains fucntion values for all the unrateed items
        if c_f == 'powe':
            xf = np.dot(np.power(Tsum_W,0.01),mind_R) +eps#xf contains fucntion values for all the unrateed items
        if c_f == 'pow1':
            xf = np.dot(np.power(Tsum_W,0.1),mind_R) +eps#xf contains fucntion values for all the unrateed items
        if c_f == 'pow9':
            xf = np.dot(np.power(Tsum_W,0.9),mind_R) +eps#xf contains fucntion values for all the unrateed items
        if c_f == 'sqrt':
            xf = np.dot(np.sqrt(Tsum_W),mind_R) +eps#xf contains fucntion values for all the unrateed items
        if c_f == 'log':
            xf = np.dot(np.log(1+Tsum_W),mind_R) + eps
        if c_f == 'riska1':
            xf = np.dot(1-np.exp(-1*Tsum_W),mind_R) + eps
        if c_f == 'riska5':
            xf = np.dot(1-np.exp(-0.5*Tsum_W),mind_R) + eps
        if c_f == 'riska01':
            xf = np.dot(1-np.exp(-0.1*Tsum_W),mind_R) + eps
        if c_f == 'riskae':
            xf = np.dot(1-np.exp(-0.001*Tsum_W),mind_R) + eps
        if c_f == 'risks':
            xf = -(np.dot(1-np.exp(1*Tsum_W),mind_R) + eps)
    #logger.info('Finished the submodular recommendation algorithm')
    return np.array(Rec_Movies_ID,dtype=int)    #this gives sorted list
###########################################################################################
def modular(Unrated_M,SiM,m_ind,mind_R,K): #greedy and b_greedy are in fact same.
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    logger.info('Modular algorithm')
    Rec_Movies_ID = list() #copy of the already rated movie indexes
    sum_W = np.zeros((Unrated_M.shape[0],m_ind.shape[0]))
    for i in range(K):  #iterate over the set of unrated movies.
        Tsum_W = sum_W + SiM[Unrated_M[:,None],m_ind]
        xf = np.dot(Tsum_W,mind_R) #xf contains fucntion values for all the unrateed items
        ind = np.argmax(xf)
        sum_W = Tsum_W[ind,:]
        best_select = Unrated_M[ind]
        Rec_Movies_ID.append(best_select) #We add the best selected to the array.
        Unrated_M  = np.delete(Unrated_M,ind)   #this is ok, since indexes are unique
    logger.info('Finished the modular version of the algorithm')
    return np.array(Rec_Movies_ID,dtype=int),0    #this gives sorted list
###########################################################################################
def mf(Unrated_M,SiM,m_ind,mind_R,K,u_ind,Rhat,MG_Mat):
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    logger.info('Simple MF algorithm')
    Rec_Movies_ID = list() #copy of the already rated movie indexes
    indxs = np.argsort(Rhat[u_ind,Unrated_M])[::-1][:K]
    logger.info('Finished simple MF version of the algorithm')
    return Unrated_M[indxs], 0
