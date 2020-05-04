#!/usr/bin/python2
from os.path import isfile
from sklearn.metrics.pairwise import cosine_similarity,rbf_kernel,linear_kernel,sigmoid_kernel
import logging
from pandas import read_csv
import numpy as np

logger = logging.getLogger(__name__)

dataset_dtls = {
                'dataset1':{
                    'folder_path'   : '../Datasets/Artificial/',
                    'ratings_file'  : 'ratings.dat',
                    'movies_file' : 'movies.dat',
                    'genres'    : ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                    'users' :[0],
                    },
                'dataset2':{
                    'folder_path'   : '../Datasets/MovieLens/',
                    'ratings_file' :   'ratings.dat',
                    'rat_columns'   : [0,1,2],
                    'delim' : '::',
                    'movies_file' : 'movies.dat',
                    'users_file'  : 'users.dat',
                    'genres'    : ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                    },
                'dataset3':{
                    'folder_path'   : '../Datasets/Ymovies/',
                    'ratings_file' :   'ratings_g15.dat',
                    'movies_file' : 'movies_g15.dat',
                    'genres'    :   ['Action', 'Adult Audience', 'Adventure', 'Animation', 'Art', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'Gangster', 'Horror', 'Kids', 'Miscellaneous', 'Musical', 'Performing Arts', 'Reality', 'Romance', 'Science Fiction', 'Special Interest', 'Suspense', 'Thriller', 'Western'],
                    },
                'dataset4':{
                    'folder_path'   : '../Datasets/ml-20m/',
                    'ratings_file' :   'ratings.csv',
                    'rat_columns' : [0,1,2],
                    'mov_columns' : [0,1,2],
                    'delim' :   ',',
                    'movies_file' : 'movies.csv',
                    'genres'    : ['Film-Noir', 'Action', 'Drama', 'Horror', 'Adventure', 'Mystery', 'War', 'Musical', 'Sci-Fi', 'IMAX', 'Documentary', 'Western', 'Animation', 'Fantasy', 'Crime', 'Children', 'Romance', 'Thriller', 'Comedy'],
                    'r_skip'    : 1,
                    'm_skip'    : 1,
                    }
                }

###########################################################################################
def get_folder_path(dataset):
    return dataset_dtls[dataset]['folder_path']
###########################################################################################
def get_genres(dataset):
    return dataset_dtls[dataset]['genres']
###########################################################################################
def get_log_file(dataset):
    return dataset_dtls[dataset]['folder_path'] + "access_%d.log" %(source,target,cnt)
###########################################################################################
def get_ratings_df(dataset):
    ratings_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['ratings_file']
    columns = dataset_dtls[dataset]['rat_columns']
    delim = dataset_dtls[dataset]['delim']
    r_skip =  dataset_dtls[dataset]['r_skip'] if 'r_skip' in dataset_dtls[dataset] else None
    if isfile(ratings_file):
        print("Reading from the ratings file %s" %(ratings_file))
        df = read_csv(ratings_file, header = 0, delimiter=delim, names=['UID','IID','RATING'], usecols=columns,quotechar='"', dtype={'UID':np.int32, 'IID':np.int32, 'RATING':np.float16}, engine='python')
        df['R'] = 0
        df.loc[df['RATING'] >3, 'R'] = 1 
        df = df[df['R'].ge(1).groupby(df['UID']).transform('sum').ge(100)]   #consider users who has more than 100 relevant ratings
        df.sort_values(by=['UID'],inplace=True)
        df.reset_index(drop=True,inplace=True)
    return df #Index of the u_id/m_id in U_map/M_map is the position in r_matrix.which(U_map==ID)
###########################################################################################
def get_rating_matrix(df):
    U_map,U_inv = np.unique(df.UID.values,return_inverse=True)
    M_map,M_inv = np.unique(df.IID.values,return_inverse=True)
    R = np.zeros((U_map.shape[0],M_map.shape[0]), dtype=np.float16)
    R[U_inv,M_inv] = df['RATING']
    return R
###########################################################################################
def split_train_test(df):
    tr_indx = (df[df.R ==1].groupby('UID',group_keys=False).apply(lambda x: x.sample(frac=0.7,axis=0,random_state=1000))).index
    return df.loc[tr_indx], df.loc[df.index.difference(tr_indx, sort=False)]
###########################################################################################
def write_to_file(df,fname):
    gd = df.groupby('UID',group_keys=False)
    with open(fname,'w') as fd:
        for k,gp in gd:
            line = str(k) + ' ' + gp.IID.to_string(index=False,header=False).replace("\n",'').strip() + "\n"
            fd.write(line)
###########################################################################################
def get_movies_df(dataset):
    movies_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['movies_file']
    columns = dataset_dtls[dataset]['mov_columns']
    delim = dataset_dtls[dataset]['delim']
    m_skip =  dataset_dtls[dataset]['r_skip'] if 'r_skip' in dataset_dtls[dataset] else None
    if isfile(movies_file):
        print("Reading from the movie file %s" %(movies_file))
        df = read_csv(movies_file, header = 0, delimiter=delim, names=['IID','TITLE','GENRE'], usecols=columns,quotechar='"', dtype={'IID':np.int32, 'TITLE':str, 'GENRE':str}, engine='python')
        df = df.join(df['GENRE'].str.get_dummies())
        df.reset_index(drop=True,inplace=True)
    return df
###########################################################################################
def filter_movies(dr,dm):
    dm = dm[dm.IID.isin(dr.IID.unique())]
    dm.sort_values(by=['IID'],inplace=True)
    dm.reset_index(drop=True,inplace=True)
    return dm
###########################################################################################
def get_test_users(R_test):
    return np.where(R_test.sum(1)!=0)[0]    #list of users who has at least one rating in the data
###########################################################################################
def get_unrated_movies(R_test,u_ind):
    #return np.where(R_test[u_ind,:] != 0)[0]
    return np.where(R_test[u_ind,:] != 0)[0],len(np.where(R_test[u_ind,:] > 3)[0])
###########################################################################################
def read_userlist(dataset):
    if 'users' in dataset_dtls[dataset].keys():
        return dataset_dtls[dataset]['users']
    else:
        return None
###########################################################################################
def get_train_test(R,te_indx):    #split the data into training and test by random sampling
    U,I = R.shape
    R_test = np.zeros(shape=(U,I),dtype=np.float16)
    R_train = np.copy(R)
    if len(te_indx) == 0:
        return R_train,R_test
    Indx = np.unravel_index(te_indx,(U,I))
    R_test[Indx] = R[Indx]
    R_train[Indx] = 0
    return R_train,R_test
#########################################################################################
def get_item_item_sim(V):  #get the
    #D = cosine_similarity(V)
    #D = rbf_kernel(V)
    #D = linear_kernel(V)
    D = sigmoid_kernel(V)
    return D
###########################################################################################
def get_predicted_ratings(U,V,max_rat):
    Rhat = np.dot(U,V)
    #W = float(max_rat)/np.max(Rhat,axis=1)[:,None].astype(float)
    #Rhat -= np.min(Rhat)
    #Rhat = Rhat*W
    return Rhat
#########################################################################################
def get_rated_movies(R,u_ind):   #Given a user indexd it retrieves the movies and ratings by the user.
    m_ind = R[u_ind,:].nonzero()[0]    #get index of already rated movies by user at u_ind. A single Row
    m_rat = R[u_ind,m_ind] #get the ratings of the movies already rated by user at u_ind
    return m_ind,m_rat  #return the already rated movie indexes and corresponding ratings of user id
###########################################################################################
def read_from_file(f_name): #we will read into a dictionary
    D = {}
    with open(f_name) as fd:
        for line in fd:
            flds = line.split(' ')
            uid = int(flds.pop(0))
            D[uid] = dict(map(float, d.split(':')) for d in flds)
    return D
###########################################################################################
def get_items_from_df(u_ind,te_df):
    return te_df.loc[(te_df.UID == u_ind) & (te_df.R == 1),'IID'].tolist() #get only relevant items. unobserved are irrelevant
