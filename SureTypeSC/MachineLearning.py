import os
import logging
from DataLoader import Data,Patterns

import Settings
import sys
from sklearn import mixture
import numpy as np
from collections import Counter
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.covariance import MinCovDet
import datetime
import Config
#import pickle
import cPickle
#import TrioLoader
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import copy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import copy
import random
from sklearn import tree
from sklearn import linear_model
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn import mixture
import protocol
from sklearn import metrics
from sklearn import linear_model

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import Imputer


from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GMM,BayesianGaussianMixture,GaussianMixture
from scipy.misc import logsumexp


#import joblib as jb


def test(x):
    return x

'''
functions create GMM classifier from dataframe
param d has cols:
"output" - true genotype (1 or 0)
"m" - logarithmic difference of signals
"a" - logarithmic average of signals
"gtype" - genotype
"pred" - prediction of 1st layer - usually random forest

returns GMM classifier
'''
'''
def _print(self, msg, msg_args):
    """ Display the message on stout or stderr depending on verbosity
    """
    # XXX: Not using the logger framework: need to
    # learn to use logger better.
    if not self.verbose:
        return
    if self.verbose < 50:
        writer = sys.stderr.write
    else:
        writer = sys.stdout.write
    msg = msg % msg_args
    writer('[%s]: %s\n' % (self, msg))
    logging.info('[%s]: %s\n' % (self, msg))
'''


def  create_components_for_gmm_roc(d,threshold,inn):
    lookup="proba_" + Config.D["CLFNAME"]
    d=d[["gtype",lookup]+Config.D["INPUT_FEATURE"]]
    d[inn]=d[inn].replace(to_replace=[np.inf, -np.inf],value=np.nan)
    d.dropna(inplace=True)
    #d.loc[(~np.isfinite(d)) & d.notnull()] = np.nan

    #covariancemat=d.groupby(["gtype","pred"])["m","a"].cov()

    #positive_class=d[d[lookup] >= threshold]
    #negative_class=d[d[lookup] < threshold]

    positive_class=(d[d[lookup] >= threshold],1)
    negative_class=(d[d[lookup] < threshold],0)




    #means_positive=positive_class.groupby(["gtype"])[Config.D["INPUT_FEATURE"]].mean()
    #means_negative=negative_class.groupby(["gtype"])[Config.D["INPUT_FEATURE"]].mean()

    #this is really just to make it go through
    #if np.inf in means.values or -np.inf in means.values or np.nan in means.values:
    #   print "substituting 0 to means - this should not happen and would bias the analysis!"
    #   means=means.replace([np.inf, -np.inf], np.nan).replace(np.nan,0.0)

    #covariancemat.index=pd.MultiIndex.from_tuples([("_".join([str(i[0]),str(i[1])]),i[2]) for i in covariancemat.index.values],names=["component","feature"])

    #covariancemat.fillna(method='backfill',inplace=True)
    #covariancemat=covariancemat.swaplevel()

    #if np.inf in covariancemat.values or -np.inf in covariancemat.values or np.nan in covariancemat.values:
    #    covariancemat=covariancemat.replace([np.inf, -np.inf], np.nan).replace(np.nan,0.0)
    covariance_ar=[]
    weights_ar=[]
    means_ar=[]
    groups=[]
    #for i in range(covariancemat.to_panel().values.shape[1]):
    #    print np.linalg.pinv(covariancemat.to_panel().values[:,i])
    #    test_ar.append(np.linalg.pinv(covariancemat.to_panel().values[:,i]))



    ####THIS IS ORIGINAL VERSION

    for d_cl in [positive_class,negative_class]:
      for g,df in d_cl[0].groupby(["gtype"]):
            covariance_ar.append(np.linalg.pinv((df[inn].cov()).values))
            weights_ar.append(len(df))
            means_ar.append(df[inn].mean())
            groups.append((g,d_cl[1]))
        #return tuple of: (means,precision_matrix,weights)
    covariance_ar=np.array(covariance_ar)
    #covariance_ar[np.isnan(covariance_ar)]=1.0
    #previously we initialised everything
    #gmm=mixture.GaussianMixture(warm_start=True,n_components=6, covariance_type='full',means_init=means.values ,precisions_init=covariance_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]))
    #gmm=mixture.GaussianMixture(n_components=6, precisions_init=covariance_ar,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=10000,verbose=2)
    gmm=mixture.GaussianMixture(n_components=6, precisions_init=covariance_ar,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=1000,verbose=0)
    #gmm=mixture.GaussianMixture(n_components=6,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=10000,verbose=2)
    ##################################

    ##This is THE adjusted version with only 4 components
    '''
    for gpos,pos_df in positive_class.groupby(["gtype"]):
        covariance_ar.append(np.linalg.pinv((pos_df[Config.D["INPUT_FEATURE"]].cov()).values))
        means_ar.append(pos_df[Config.D["INPUT_FEATURE"]].mean())
        weights_ar.append(len(pos_df))
        groups.append(gpos)


    covariance_ar.append(np.linalg.pinv((negative_class[Config.D["INPUT_FEATURE"]].cov()).values))
    means_ar.append(negative_class[Config.D["INPUT_FEATURE"]].mean())
    weights_ar.append(len(negative_class))
    groups.append('Null')


    #gmm.means_=means_ar
    #gmm.precisions_=covariance_ar
    #gmm.weights_=np.array([float(i)/sum(weights_ar) for i in weights_ar])
    gmm=mixture.GaussianMixture(n_components=4, covariance_type='full',means_init=np.array(means_ar) ,precisions_init=np.array(covariance_ar),weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]))
    '''

    return (groups,gmm)


class GMMcl:
    n_components = None
    covariance_type = None
    nClasses = None
    mixtures = None

    def __init__(self,core,n_components=None, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.core=core


    def fit(self, X,Y):
        self.classes_=np.unique(Y)
        self.group_weights=[len(X[Y==i]) / float(len(X)) for i in self.classes_]
        #self.group_weights=self.groups.apply(len)/sum(self.groups.apply(len))
        self.group_log_weights=np.log(self.group_weights)
        self.nClasses = len(self.classes_)
        #self.mixtures = [BayesianGaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=2,verbose_interval=100)
        if self.core=="gmm":
          self.mixtures = [GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=0)
                         for k in range(self.nClasses)]
        else:
          self.mixtures = [BayesianGaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=0)
                         for k in range(self.nClasses)]
        #self.mixtures = [GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=2,verbose_interval=100)
        #                 for k in range(self.nClasses)]
        for g,mix in zip(self.classes_, self.mixtures):
           mix.fit(X[Y==g])
        #print self.mixtures.
        return self

    def predict(self, X):
        ll = [ mix.score_samples(X)for k, mix in enumerate(self.mixtures) ]
        #ll = [ mix.score(X[["m","a"]])for k, mix in enumerate(self.mixtures) ]
        #return np.vstack(ll).argmax(axis=0)
        return np.take(self.classes_, np.vstack(ll).argmax(axis=0), axis=0)

    def predict_proba(self,X):
        ll = [ mix.score_samples(X)for k, mix in enumerate(self.mixtures) ]
        #ll = [ mix.score(X[["m","a"]])for k, mix in enumerate(self.mixtures) ]
        #for ind,val in enumerate(self.groups.groups.keys()):
        #    if val==1: positive_class=ind
        #    else: negative_class=ind
        positive_class=np.where(self.classes_ == 1)[0][0]
        negative_class=np.where(self.classes_ == 0)[0][0]

        #this was not in the original implementation, should have an adaptive role
        binary_predictions=self.predict(X)
        self.group_weights=[len(X[binary_predictions==i]) / float(len(X)) for i in self.classes_]
        self.group_log_weights=np.log(self.group_weights)

        #normalize by class weight

        ##TODO recalculate group_weights


        ll_norm=np.vstack(ll)+ np.vstack(self.group_log_weights)
        #return np.vstack(ll).argmax(axis=0)
        #return np.vstack(ll)[positive_class]
        #return np.exp((ll_norm-logsumexp(ll_norm,axis=0))[positive_class])
        return np.exp((ll_norm-logsumexp(ll_norm,axis=0))).transpose()




class TrainingDataCreator(object):
    def __init__(self,query,exclude_chr=[]):
        '''

        :param query:     type Data from DataLoader
        :return:
        '''
        #print(out)
        self._query=query
        #self._input_feat=inn
        #self._output_feat=out
        self._exclude_chr=exclude_chr

    @staticmethod
    def _run_outliers(x):
         x['outliers'] = MinCovDet().fit(x[['m_raw',"a_raw"]]).support_
         return x

    def create(self,type="all",outliers=True,jobid=None,liers=False,mode="original",inn=['1','2'],out="output",ratio=1):
        '''
        default behaviour is to create
        :return:
        '''

        input_feat=inn
        output_feat=out
        exclude_chr=self._exclude_chr


        levels_to_stack=[i for i in self._query.df.columns.names if i!="feature"]

        t=self._query.df.stack(levels_to_stack)


        t = t.replace([np.inf, -np.inf], np.nan)

        ###t.dropna(inplace=True)
        #this assures chr is part of the input feat
        t.reset_index(level="Chr",inplace=True)
        t['Chr']=t['Chr'].astype('category')

        #t['gtype']=t['gtype'].astype('category')
        t=t[(t['gtype']!="NC") & (~t.Chr.isin(exclude_chr))]
        if type!="all":
            if type=="hom":
              t=t[(t['gtype']=="AA") | (t['gtype']=="BB")]
            elif type=="het":
               t=t[t['gtype']=="AB"]

        logging.info('Training Dataset properties' + str(Counter(t['gtype'])))
        if jobid:
          protocol.add_stat(jobid,str(Counter(t['gtype'])))
        else:  protocol.send_data('Dataset properties' + str(Counter(t['gtype'])))


        t=t[input_feat+[output_feat]].dropna()

        t[output_feat]=t[output_feat].astype(int)
        
        #print liers
        if liers:
          t=t.groupby(["Chr",'gtype']).apply(self._run_outliers)

        #print 't',t

        if 'gtype' in input_feat:
            #t['gtype']=t['gtype'].astype('category')
            t['gtype'].replace(Settings.Settings.CODE, inplace=True)
            t['gtype']=t['gtype'].astype('category')




        X=pd.get_dummies(t[input_feat]).values
        #print X
        if liers:
          #Y=pd.get_dummies(((t.output.apply(int) + t.outliers.apply(int)) > 1).apply(int))
          Y=((t.output.apply(int) + t.outliers.apply(int)) > 1).apply(int)
        else:
          Y=t[output_feat].apply(int)
          #print output_feat
          #print t[output_feat]
        #X=t[input_feat].values
        #Y=t[output_feat].values


        ####
        #here we use downsampling if needed
        ####
        nsamples=len(X)
        noutputs=len(Y)
        if nsamples != noutputs:
            raise ValueError('#samples != #outputs')




        #indices for all samples
        indall=range(0,nsamples)

        #number of positive samples
        npositive=sum(Y.values)
        print('positive',npositive)

        #number of negative samples
        nnegative=len(Y.values)-npositive
        print('nnegative',nnegative)

        #if npositive<nnegative:
        #    raise ValueError('#positive<#negative in the input dataset!')



        cur_ratio=npositive/(nnegative*1.0)


        #ratio=Config.D["RATIO"]

        if ratio==0:#keep the original dataset
           final_ind=indall
        else:
           #indices for positive samples
           indpositive=np.argwhere(Y.values==1)[:,0]
           #indices for negative samples
           indnegative=np.argwhere(Y.values==0)[:,0]

           nsamplepos=np.floor(npositive/(cur_ratio/(ratio*1.0)))
           #else: nsamplepos=np.floor(npositive*(cur_ratio/(ratio*1.0)))
           nsampleneg=nnegative

           selindpos=np.random.choice(indpositive,int(nsamplepos))
           selindneg=indnegative

           final_ind=np.concatenate((selindpos,selindneg))

        positive=sum(Y[final_ind])
        negative=len(Y[final_ind])-positive

        #logging.info("Using training ratio {}".format(positive/(negative*1.0)))
        #logging.info("Dataset: " + Config.D["ID"] +" positive:" + str(positive) + ", negative: " + str(negative))

        ###############
        return (X[final_ind],Y.values[final_ind])

    #method creates training data for
    def create_components(self,type="all",outliers=True,jobid=None,inn=['m','a'],out='output'):
        '''
        default behaviour is to create
        :return:
        '''

        input_feat=inn
        output_feat=out
        exclude_chr=self._exclude_chr
        levels_to_stack=[i for i in self._query.df.columns.names if i!="feature"]
        t=self._query.df.stack(levels_to_stack)


        t = t.replace([np.inf, -np.inf], np.nan)

        ###t.dropna(inplace=True)
        #this assures chr is part of the input feat
        t.reset_index(level="Chr",inplace=True)
        t['Chr']=t['Chr'].astype('category')

        #t['gtype']=t['gtype'].astype('category')
        t=t[(t['gtype']!="NC") & (~t.Chr.isin(exclude_chr))]
        if type!="all":
            if type=="hom":
              t=t[(t['gtype']=="AA") | (t['gtype']=="BB")]
            elif type=="het":
               t=t[t['gtype']=="AB"]

        #tu by mala byt statistika!!!

        X=pd.get_dummies(t[input_feat].dropna()).values

        logging.info('Dataset properties' + str(Counter(t['gtype'])))
        if jobid:
          protocol.add_stat(jobid,str(Counter(t['gtype'])))
        else:  protocol.send_data('Dataset properties' + str(Counter(t['gtype'])))


        #covariancemat=t.groupby(["gtype","output"])["m","a"].cov().apply(lambda x: pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index) )
        #covariancemat=t.groupby(["gtype","output"])["m","a"].agg(lambda x:pd.DataFrame(np.linalg.pinv(x.cov()),x.cov().columns, x.cov().index ))



        #covariancemat=t.groupby(["gtype","output"])["m","a"].cov()
        #covariancemat=t.groupby(["gtype","output"])[Config.D["INPUT_FEATURE"]].cov()

        #means=t.groupby(["gtype","output"])[Config.D["INPUT_FEATURE"]].mean()

        #covariancemat.index=pd.MultiIndex.from_tuples([("_".join([i[0],str(i[1])]),i[2]) for i in covariancemat.index.values],names=["component","feature"])
        #covariancemat=covariancemat.swaplevel()



        covariance_ar=[]
        weights_ar=[]
        groups=[]
        means=[]
        #for i in range(covariancemat.to_panel().values.shape[1]):
        #    print np.linalg.pinv(covariancemat.to_panel().values[:,i])
        #    test_ar.append(np.linalg.pinv(covariancemat.to_panel().values[:,i]))


        for g,df in t.groupby(["gtype","output"]):
            covariance_ar.append(np.linalg.pinv((df[inn].cov()).values))
            weights_ar.append(len(df))
            groups.append(g)
            means.append(df[inn].mean())
        #return tuple of: (means,precision_matrix,weights)


        ###Adjusted version
        '''
        for g,df in t.groupby("output"):
            if g:
               for g2,df2 in df.groupby("gtype"):
                 covariance_ar.append(np.linalg.pinv((df2[Config.D["INPUT_FEATURE"]].cov()).values))
                 means.append(df2[Config.D["INPUT_FEATURE"]].mean())
                 weights_ar.append(len(df2))
                 groups.append(g2)
            else:
                covariance_ar.append(np.linalg.pinv((df[Config.D["INPUT_FEATURE"]].cov()).values))
                means.append(df[Config.D["INPUT_FEATURE"]].mean())
                weights_ar.append(len(df))
                groups.append('Null')
        '''
        ####################


        #return (X,means.values,np.array(covariance_ar),np.array([float(i)/sum(weights_ar) for i in weights_ar]),groups)
        return (X,np.array(means),np.array(covariance_ar),np.array([float(i)/sum(weights_ar) for i in weights_ar]),groups)


    def add_outliers(self,func=MinCovDet):
        pass
        #self._query.detect_outliers(func,self._input_feat)


class Trainer(object):
    def __init__(self,data,clfname='gda',inner=['m','a'],outer='rf_ratio:1.0_pred',jobid=None,trees=30,outlie=False,mode="original",ratio=1):

      self._samples=data.get_samples_names()
      logging.info('Samples in the bucket:' + str(self._samples))
      if jobid:
          protocol.add_stat(jobid,sorted(self._samples))

      if clfname=="gaussian":
         #self._X,self._Y=TrainingDataCreator(data).create(outliers=Config.D["OUTLIERS"],type="het")
         #self._clf=mixture.GaussianMixture(n_components=2, covariance_type='full',means_init=[[np.mean(self._X[self._Y==0])] ,[np.mean(self._X[self._Y==1])]])
         self._X,self._Means,self._Y,weights,self._components=TrainingDataCreator(data).create_components(outliers=out,type="all",jobid=jobid)
         self._clf=mixture.GaussianMixture(n_components=len(weights), covariance_type='full',means_init=self._Means ,precisions_init=self._Y,weights_init=weights,verbose=0)

         #self._clf=mixture.GaussianMixture(n_components=6, covariance_type='full',verbose=2)

         #self._clf=mixture.GaussianMixture(n_components=6, covariance_type='full',means_init=self._Means,precisions_init=self._Y)
         #self._clf=mixture.GaussianMixture(n_components=5, covariance_type='full',means_init=self._Means,precisions_init=self._Y)
      else:
         #self._X,self._Y=TrainingDataCreator(data).create(outliers=Config.D["OUTLIERS"],type=Config.D["TRAINING_SELECTION"],jobid=jobid)
         self._X,self._Y=TrainingDataCreator(data).create(liers=outlie,type="all",jobid=jobid,inn=inner,out=outer,mode=mode,ratio=ratio)
       #self._clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
      if clfname=='rf':
         #self._clf=RandomForestClassifier(n_jobs=-1,n_estimators=128,max_features=None)
         self._clf=RandomForestClassifier(n_jobs=-1,n_estimators=trees,max_features=None)
      elif clfname=='dt':
         self._clf=tree.DecisionTreeClassifier()
      elif clfname=='kde':
           self._clf = (KernelDensity(kernel='gaussian'),KernelDensity(kernel='gaussian'))
      elif clfname=="lr":
        self._clf= linear_model.LogisticRegression(n_jobs=-1)
      elif clfname=="knn":
         self._clf=KNeighborsClassifier(n_jobs=-1)
      elif clfname=="bgmm":
         self._clf=mixture.BayesianGaussianMixture(n_components=7,weight_concentration_prior=0.01)
      elif clfname=="mlp":
         self._clf= MLPClassifier()
      elif clfname=="vbgmmcl" or "vbgmmcl" in clfname:
          self._clf=GMMcl(n_components=3,core="vbgmm")
      elif clfname=="gmmcl" or "gmmcl" in clfname or "gda" in clfname:
          self._clf=GMMcl(n_components=3,core="gmm")
      

       #self._clf=SVC()
       #self._clf=KNeighborsClassifier(n_jobs=-1)


       #self._names=["BB_False","AA_True","BB_True","AB_True","AA_False",]
       #self._names=["BB_True","BB_False","AB_True","AA_False","AA_True"]

    '''this method is for GaussianMixture

    def __train(self):
       self._clf.fit(self._X)
       t=sorted([i for i in enumerate(self._clf.means_)], key=lambda x: x[1])
       order=sorted([(ord,i[0]) for ord,i in enumerate(t)],key=lambda x:[x[1]])
       # t is [(ord,val)]
       self._ordered_names=[self._names[o[0]] for o in order]
       return self._clf
    '''
 
    def train(self,name='rf-gda'):
        if name =='kde':
          self._clf[0].fit(self._X[self._Y==1])
          self._clf[1].fit(self._X[self._Y==0])
        elif name =='gaussian':
          #self._clf.fit(self._X)
          self._clf.fit(self._Means)

        #elif Config.D["CLFNAME"]=="vbgmmcl":
        #  self._clf.fit(self._X,groupvar="pred")

        else:
          self._clf.fit(self._X,self._Y)



    def predict_self(self):
        return self._clf.predict(self._X)

    def predict_self_proba(self):
        return self._clf.predict_proba(self._X)

    def predict(self,X,name):
        if name=='kde':
            pos=self._clf[0].score_samples(X)
            neg=self._clf[1].score_samples(X)
            return np.array([ 1 if x>0.6 else 0 for x in pos/(pos+neg)])

        elif name=="gaussian":
          return np.array([int(self._components[i][1]) for i in self._clf.predict(X)])
        else:
          return self._clf.predict(X)



    #to make this universal, GMM should also do binary classification
    def predict_proba(self,X,name):
        if name =='kde':
            pos=self._clf[0].score_samples(X)
            neg=self._clf[1].score_samples(X)
            return np.array([ 1 if x>0.6 else 0 for x in pos/(pos+neg)])

        elif name=="gaussian":
          return np.array([int(self._components[i][1]) for i in self._clf.predict_proba(X)])

        elif name=="bgmm":
          pass

        else:
          positive_class=np.where(self._clf.classes_==1)
          #return self._clf.predict_proba(X)[positive_class]
          return (self._clf.predict_proba(X)[:,positive_class]).flatten()


    #this methods takes only 1
    def create_roc_data_one_layer(self,test,inn,clftype,selection='all'):
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        if selection=="all":
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')


        if clftype!="gaussian":
          for ind,val in enumerate(self.clf.classes_):
            if val==1: positive_class=ind
          stacked["_".join(["proba", clftype])]=self.predict_proba(pd.get_dummies(stacked[inn]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))[:,positive_class]
        else:
          gmm_pred=self.clf.predict_proba(stacked[inn].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))
          #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(self._components)!="Null"),axis=1).sum(axis=1)
          positive_gmm=np.take(gmm_pred,np.argwhere(np.array(self._components)[:,1]),axis=1).sum(axis=1)
          stacked["_".join(["proba", clftype])]=positive_gmm
          #component_ind=np.argwhere(np.array(self._components)[:,1]==1)[np.array(stacked["gtype"])]

          #probabilities of the gtype from GMM
          #probs_gmm=np.take(gmm_pred,component_ind[:,0])


        return stacked



    def create_roc_data(self,test,inn,clftype,selection='all'):
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        if selection=="all":
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')

        for ind,val in enumerate(self.clf.classes_):
            if val==1: positive_class=ind

        lookup="_".join(["proba", clftype])


        X=Imputer().fit_transform(pd.get_dummies(stacked[inn].replace([np.inf, -np.inf], np.nan)))


        #stacked[lookup]=self.predict_proba(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))[:,positive_class]
        stacked[lookup]=self.predict_proba(X)[:,positive_class]



        #we create GMM from values that are for sure true according to the classifier - probability 1.0
        # ... or for sure false - probability 0.0
        ##(components,gmm)=create_components_for_gmm_roc(stacked.query("proba_rf==1 or proba_rf==0")[["proba_rf","gtype","output"] + Config.D["INPUT_FEATURE"]])
        (components,gmm)=create_components_for_gmm_roc(stacked[[lookup,"gtype","output"] + inn],threshold=0.5)

        #in the previous approach we only used means, lets try it again
        ##gmm.fit(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).dropna())

        #gmm.fit(X)
        gmm.fit(gmm.means_init)

        #gmm.fit(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))


        #stacked["pred_gmm"]=gmm.predict_proba(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))
        gmm_pred=gmm.predict_proba(X)

        gmm_bin_pred=gmm.predict(X)

        #np.savetxt("foo.csv", gmm_pred, delimiter=",")
        #print "saved posterior prob..."
        #get indices of the components referring to positive clusters
        #component_ind=np.argwhere(np.array(components)[:,1]==1)[np.array(stacked["gtype"])]

        #probabilities of the gtype from GMM
        #probs_gmm=np.take(gmm_pred,component_ind[:,0])



        #stacked=stacked.assign(pred_gmm_component=np.array(components)[np.argmax(gmm_pred,axis=1)],pred_gmm=np.max(gmm_pred,axis=1))
        stacked=stacked.assign(pred_gmm_component=np.array(components)[np.argmax(gmm_pred,axis=1)][:,0],pred_gmm=np.max(gmm_pred,axis=1))
        #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)!="Null"),axis=1).sum(axis=1)
        #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)!='Null'),axis=1).sum(axis=1)
        positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)[:,1]==1),axis=1).sum(axis=1)


        #this is the yscore
        #stacked["proba_gmm"]=probs_gmm
        stacked["gmm_bin"]=gmm_bin_pred
        #stacked["proba_gmm_rf"]=probs_gmm*stacked["proba_rf"].values
        stacked["proba_gmm_"+ clftype]=np.maximum(positive_gmm[:,0],stacked[lookup].values)
        stacked["proba_positive_gmm_" + clftype]=positive_gmm[:,0]*stacked[lookup]
        stacked["positive_gmm"]=positive_gmm


        #this is the correct output

        #gmm.predict_proba(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))



        #return stacked[["gtype","output","gmm_bin","proba_" + Config.D["CLFNAME"],"proba_gmm","positive_gmm","proba_gmm_"+ Config.D["CLFNAME"],"proba_positive_gmm_"+ Config.D["CLFNAME"],"score"] + Config.D["INPUT_FEATURE"]]
        return stacked

    def predict_decorate(self,test,clftype='rf',inn=['m','a'],threshold=0.66,selection='all',ratio=1.0):
        print 'classifier type',clftype
        #stacked=test.df.stack("individual").reset_index(level="Chr")
        #this is more generic version:00001 = {float64} 0.869273134945
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        #if Config.D["SELECTION"]=="all":
        if selection == 'all':
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')

        rec_name="{0}_ratio:{1}_pred".format(clftype,ratio)
        rec_name_prob="{0}_ratio:{1}_prob".format(clftype,ratio)

        stacked[rec_name]=self.predict(pd.get_dummies(stacked[inn]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0),clftype)
        stacked[rec_name_prob]=self.predict_proba(pd.get_dummies(stacked[inn]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0),clftype)

        stacked[rec_name_prob]=stacked[rec_name_prob].astype(float)


        #if "threshold" in kwargs.keys():
        stacked[rec_name]=[1 if i>threshold else 0 for i in stacked[rec_name_prob]]
        #else:
        #    stacked["pred"]=self.predict(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))


        #####load mixture model
        '''
        if Config.D["CLFNAME"]!="gaussian":
            gmm=mixture.GaussianMixture(n_components=2, covariance_type='full',means_init=[[np.mean(stacked[stacked.pred==0].a)] ,[np.mean(stacked[stacked.pred==1].a)]])
            gmm.fit(stacked[["a"]])
            stacked["pred_gauss"]=gmm.predict(stacked[["a"]])
        '''
        #Y_pred_gaus=gmm.predict([X_test[:,[1]]])


        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        #stacked=stacked.set_index("Chr", drop=True, append=True).unstack(level=levels_to_stack).swaplevel(1,0,axis=1).swaplevel(2,1,axis=0)
        stacked=stacked.set_index("Chr", drop=True, append=True).unstack(level=levels_to_stack).reorder_levels(test.df.columns.names,axis=1).swaplevel(2,1,axis=0)



        stacked.sortlevel(axis=0,inplace=True,sort_remaining=True)
        stacked.sortlevel(axis=1,inplace=True,sort_remaining=True)

        '''
        if Config.D["CLFNAME"]!="gaussian":
          #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred","pred_gauss"])]],axis=1,join="inner")
          o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred","pred_gauss"])]],axis=1)
        else:
          #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred"])]],axis=1,join="inner")
        '''
        if rec_name in test.df.columns.get_level_values(level="feature") or rec_name_prob in test.df.columns.get_level_values(level="feature"):
           #test.df.update(stacked.loc[:,(slice(None),["pred"])])
           #more generic version
           test.df.update(stacked.xs(rec_name,axis=1,drop_level=False,level="feature"))
           test.df.update(stacked.xs(rec_name_prob,axis=1,drop_level=False,level="feature"))
           #test.df.join(stacked.xs("pred",axis=1,drop_level=False,level="feature"))
        else:
            #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred"])]],axis=1)
            #more genetic version
            test.df=test.df.join(stacked.xs(rec_name,axis=1,drop_level=False,level="feature"))
            test.df=test.df.join(stacked.xs(rec_name_prob,axis=1,drop_level=False,level="feature"))
            test.df.sortlevel(axis=0,inplace=True,sort_remaining=True)
            test.df.sortlevel(axis=1,inplace=True,sort_remaining=True)
            #update
            #test.df=o
        #return Dataobject with decorated dataframe

        test.container.append(rec_name)
        test.container.append(rec_name_prob)
        return test



    @property
    def clf(self):
        return self._clf

    '''this method is again for GaussianMixture
    def __evaluate(self):
        components=self.predict_self()
        converted=[self._ordered_names[i] for i in components]
        return (Counter((converted==self._Y.T)[0])[True],Counter((converted==self._Y.T)[0])[False])
    '''

    def evaluate(self):
        pass

    def save_model(self):
        '''serialize model to file
        :return:
        '''
        with open("RF.model", "w") as f:
           print "serializing ML model..."
           pickle.dump(self._clf, f)


class SerializedTrainer(Trainer):
    def __init__(self,s_clf):
        self._clf=s_clf


class Tester(Trainer):
    def __init__(self,clfpath):
        with open(clfpath,"rb") as f:
           print "deserializing ML model..."
           self._clf = pickle.load(f)




class Validator(object):
    def __init__(self,model,data):
        '''
        Load model and testing data
        :param model:
        :param data:
        :return:
        '''
        pass


    def validate(self):
        pass



def save_state():
    gdna=Data.create_from_file(sys.argv[1],"GDNA",exclude=Settings.Settings.TO_REMOVE)
    sc=Data.create_from_file(sys.argv[2],"SC")

    sc.custom_filter("sc")
    mother=gdna.slice("gm07224")
    father=gdna.slice("gm07225")
    ref_proband=gdna.slice("gdna")
    p=Patterns.check_parental_trios(mother,father,ref_proband)
    ref_proband.add_lq_indices(list(p[p==False].index.values))
    sc.calculate_transformations()
    sc.compare_against_reference(ref_proband)
    sc.create_transition_matrices(ref_proband)

    #sc.compare_against_reference(ref_proband)


    sc.calculate_group_columns_index()


    with open("SC.obj", "w") as f:
        print "serializing SC..."
        pickle.dump(sc, f)


    with open("REF.obj", "w") as f:
        print "serializing reference gdna..."
        pickle.dump(ref_proband, f)

    print "Done..."



def deserialize_and_run():
    with open(Config.D["SCOBJECT"], 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
      ref_proband = pickle.load(f)
      print "finished loading object"
      print ref_proband.get_individuals_count()


def starting_procedure_load_sc():
    with open(Config.D["SCOBJECT"], "rb") as input_file:
      sc = cPickle.load(input_file)

    with open(Config.D["SCTRAINOBJECT"], "rb") as input_file:
      sc_train = cPickle.load(input_file)

    return (sc_train,sc)



'''
this should be generic method that returns the single cell Data object and gDNA Data object
'''
def starting_procedure():
    gdna=Data.create_from_file(Config.D["GDNA"],"GDNA",exclude=Settings.Settings.TO_REMOVE)
    logging.info("Loaded " + Config.D["GDNA"])
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")

    sc=Data.create_from_file(Config.D["SC"],"SC")
    logging.info("Loaded " + Config.D["SC"])

    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to chromosomes defined in {} ...".format(Config.D["CHROMOSOMES"]))


    #gdna.df=gdna.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #sc.df=sc.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #sc.df=sc.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #gdna.df=gdna.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    sc.custom_filter("sc")
    mother=gdna.slice("gm07224")
    father=gdna.slice("gm07225")
    ref_proband=gdna.slice("gdna")
    p=Patterns.check_parental_trios(mother,father,ref_proband)
    logging.info("Parental patterns checked...")
    ref_proband.add_lq_indices(list(p[p==False].index.values))


    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)




          #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)


def starting_procedure_T21():
    gdna=Data.create_from_file(Config.D["GDNA"],"GDNA",exclude=Settings.Settings.TO_REMOVE + ["GenTrain Score"])
    logging.info("Loaded " + Config.D["GDNA"])
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")
    #gdna.consensus_genotype()


    sc=Data.create_from_file(Config.D["SC"], "SC", exclude=["GenTrain Score"])
    ####THIS IS USUALLY 0.01
    sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=True)

    logging.info("Loaded " + Config.D["SC"])
    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to autosomes...")

    ref_proband=gdna

    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)

    #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)


def starting_procedure_GM12878():
    sc=Data.create_from_file(Config.D["SC"], "SC", exclude=["GenTrain Score"])
    logging.info("Loaded " + Config.D["SC"])
    gdna=sc.slice("gdna")
    logging.info("Loaded gDNA by slicing..." )

    sc=sc.remove("gdna")
    logging.info("removed gDNA from SC dataset" )
    ####THIS IS USUALLY 0.01
    sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=True)
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")





    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to autosomes...")

    ref_proband=gdna

    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    ######protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)

    #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)





def score_range(start, end, step):
    while start <= end:
        yield start
        start += step


def tm_routine(container,ref,sc,score,run,type):
  sc_transition_matrix,gdna_transition_matrix,raw_transition_matrix=sc.create_transition_matrices(ref)
  container[(score,run,type,"sc")]=sc_transition_matrix
  container[(score,run,type,"gdna")]=gdna_transition_matrix
  container[(score,run,type,"transition_count")]=raw_transition_matrix


def get_optimal_threshold(sbj):
  d={}
  for g,df in sbj.groupby(["type","curve"]):
    optimal_idx=np.argmax(df["tpr_recall"]-df["fpr_precision"])
    d[g]=df["threshold"][optimal_idx]
  return d



'''

'''
#def evaluate_metrics(df,func=[metrics.precision_score,metrics.recall_score,metrics.accuracy_score]):
def evaluate_metrics(df,colname,thr,names=["precision"],func=[metrics.precision_score]):
    #stack and remove na
    #colname="{}_proba".format(Config.D["CLFNAME"])
    stacked=df.stack(level=0)[["output",colname]].dropna()

    for n,f in zip(names,func):
      if n not in ["roc_auc_score"]:
        yield n,f(stacked["output"].apply(int),stacked[colname]>=thr)
      else:
        yield n,f(stacked["output"].apply(int),stacked[colname])








        
