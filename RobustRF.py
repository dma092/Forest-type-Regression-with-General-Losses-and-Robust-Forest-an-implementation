#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:26:22 2018

@author: mohamed
"""
from __future__ import division
from sklearn.ensemble import RandomForestRegressor
import warnings
import numpy as np

class RobustRandomForest(RandomForestRegressor):
    def __init__(self,n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
        
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
    
    def fit(self,IN, OUT, sample_weight=None):
       return super(RobustRandomForest,self).fit(IN, OUT, sample_weight=sample_weight)
    def predict(self,X):
       return super(RobustRandomForest,self).predict(X)
    def apply(self,X):
       return super(RobustRandomForest,self).apply(X)
    def robustPredict(self,newX,X,y):
       leaves=self.apply(newX)
       ypred=np.zeros((len(newX),1))
       observationLeaves=self.apply(X)
       self.weights=np.zeros((len(newX),len(X),self.n_estimators))
       self.sumWeights=np.zeros((len(newX),len(X)))
       try:
           for k in range(0,len(newX)):
               for i in range(0,self.n_estimators):
                   for j in range(0,len(X)):
                       if observationLeaves[j,i]==leaves[k,i]:
                           self.weights[k,j,i]=1
                   self.weights[k,:,i]=(self.weights[k,:,i]/self.weights[k,:,i].sum())
               self.sumWeights[k,:]=self.weights[k,:,:].sum(axis=1)/self.n_estimators
               ypred[k]=(self.sumWeights[k,:]*y[:]).sum()+ypred[k]
       except Exception as ex:
           print(ex)
       return(ypred,self.sumWeights)
    def robustPredictUsingHuber(self,xj,x,y,e0=10e-06,delta=0.005):
        e=100
        n=len(y)
        yhat,Wij=self.robustPredict(xj,x,y)
        k=0  
        Wijold=Wij
        yhatold=yhat
        while(e>e0):
            
            Wijnew=Wij/(np.sqrt(1+np.power((yhatold-y)/delta,2))) 
            
            yhatnew=Wijnew.dot(y)/Wijnew.sum(axis=1)
            
            
            
            e=np.mean(np.power((yhatnew.reshape(len(xj),1)-yhatold),2))
            yhatold=yhatnew.reshape(-1,1)
            k=k+1
        return(yhatnew)
    def robustPredictUsingTukey(self,xj,x,y,e0=10e-06,delta=0.8):
        
        #warnings.filterwarnings('error')
        try:
            e=100
            n=len(y)
            yhat,Wij=self.robustPredict(xj,x,y)
            k=0  
            Wijold=Wij
            yhatold=yhat
            while(e>e0):
                
                Wijnew=Wij* np.maximum(1-np.power(((yhatold-y)/delta),2),0)
                sumWeights=Wijnew.sum(axis=1)
                sumWeights[sumWeights==0]=0.1
                yhatnew=Wijnew.dot(y)/sumWeights
                e=np.mean(np.power((yhatnew.reshape(len(xj),1)-yhatold),2))
                yhatold=yhatnew.reshape(-1,1)
                k=k+1
            return(yhatnew)

        except Warning as e:
            print (e)
            return(-1)
    def KneighborstPredict(self,K,newX,X,y):
        ##Far from over
        ypred,weights=self.robustPredict(newX,X,y)
        order=len(X)-1-weights.argsort(axis=1)
        topKWeights=weights[order<=K]
        auxY=np.repeat(y,len(newX),).reshape((len(y),len(newX))).T
        #weights[order<=K]=weights[order<=K]/(weights[order<=K].reshape((len(newX),K+1)).sum(axis=1))
        weightsOrdered=weights[order<=K].reshape((len(newX),K+1))
        weightsOrdered=weightsOrdered/(weightsOrdered.sum(axis=1))
        auxY=auxY[order<=K].reshape((len(newX),K+1))
        return((weights[order<=K]*auxY[order<=K]).sum(axis=1))
        
        
     
                   
           
       
    
    
    