from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd 
from pandas import DataFrame, read_csv
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy.matlib
from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Mixture, Logistic, Laplace, PointMass, Gamma)

def train_model():
    time_scale = 15
    mmse_scale = 20
    adas_scale = 50
    cd_scale = 10
    base_age = 60
    education_scale = 20
    data = np.array(pd.read_csv('/home/zyy/Documents/kdd-master/myweb/doc-django-master/media/documents/test_data.txt'
                                , sep=",", header = None),dtype='f')
    len = np.size(data,0)
    X_test = data[0,:]
    Y1_test = data[1:len,0]
    Y2_test = data[1:len,1]
    Y5_test = data[1:len,2]
    T_test = data[1:len,3]
    X_test[2] = X_test[2]/education_scale
    
    Y1_test = np.array(Y1_test)/mmse_scale
    where_are_NaNs = np.isnan(Y1_test)
    Y1_test[where_are_NaNs] = 0

    Y2_test = np.array(Y2_test)/adas_scale
    where_are_NaNs = np.isnan(Y2_test)
    Y2_test[where_are_NaNs] = 0

    Y5_test = np.array(Y5_test)/cd_scale
    where_are_NaNs = np.isnan(Y5_test)
    Y5_test[where_are_NaNs] = 0
    Mask_T = (T_test>0)*1
    T_test = np.array(Mask_T*(T_test-base_age)/time_scale)
    Md = 5
    D = np.size(X_test)
    Ne = 1
    X_test = np.reshape(X_test,(Ne,D))
   
    a0 = np.reshape(np.array(pd.read_csv('./myapp/a0.txt', sep=",", header = None),dtype='f'),(Md))
    b0 = np.reshape(np.array(pd.read_csv('./myapp/b0.txt', sep=",", header = None),dtype='f'),(1))
    w0 = np.reshape(np.array(pd.read_csv('./myapp/w0.txt', sep=",", header = None),dtype='f'),(D,1))
    v0 = np.reshape(np.array(pd.read_csv('./myapp/v0.txt', sep=",", header = None),dtype='f'),(D,1))
    h0 = np.reshape(np.array(pd.read_csv('./myapp/h0.txt', sep=",", header = None),dtype='f'),(Md))
    c0 = np.reshape(np.array(pd.read_csv('./myapp/c0.txt', sep=",", header = None),dtype='f'),(Md))
    sigma_y0 = np.reshape(np.array(pd.read_csv('./myapp/sigma_y0.txt', 
                                               sep=",", header = None),dtype='f'),(Md))
    sigma_s0 = np.reshape(np.array(pd.read_csv('./myapp/sigma_s0.txt', 
                                               sep=",", header = None),dtype='f'),(Md))
    sigma_q0 = np.reshape(np.array(pd.read_csv('./myapp/sigma_q0.txt', 
                                               sep=",", header = None),dtype='f'),(Md))

    Mt = np.size(Y1_test)
    Y1_test = np.reshape(Y1_test,(Ne,Mt))
    Y2_test = np.reshape(Y2_test,(Ne,Mt))
    Y5_test = np.reshape(Y5_test,(Ne,Mt))
    T_test = np.reshape(T_test,(Ne,Mt))
    Mask_test = np.zeros((Md-2,Ne,Mt))
    Mask_test[0,:,:] = (Y1_test>0)*1
    Mask_test[1,:,:] = (Y2_test>0)*1
    Mask_test[2,:,:] = ((Y5_test-Y5_test)==0)*1 
    Me = np.size(Y1_test,1)
    if Me>0:
        Xt = tf.placeholder(tf.float32, [Ne,D])
        Tt = tf.placeholder(tf.float32, [Ne,Me])  
        Maskt = tf.placeholder(tf.float32, [Md-2,Ne,Me])
        Yt1 = tf.placeholder(tf.float32,[Ne,Me])
        Yt2 = tf.placeholder(tf.float32,[Ne,Me])    
        Yt5 = tf.placeholder(tf.float32,[Ne,Me])  
    # models   
        s1 = Normal(loc=(tf.matmul(Xt,w0) + b0), scale = sigma_s0[0])
        q11 = Normal(loc=tf.matmul(Xt,v0) + a0[0], scale=sigma_q0[0])
        q12 = Normal(loc=tf.matmul(Xt,v0) + a0[1], scale=sigma_q0[1])
        q15 = Normal(loc=tf.matmul(Xt,v0) + a0[4], scale=sigma_q0[4])
       
        q_s1 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))    
        q_q11 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
        q_q22 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
        q_q55 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
        
        Zp1 = tf.sigmoid(tf.abs(s1)*(Tt-q11))
        Zp2 = tf.sigmoid(tf.abs(s1)*(Tt-q12))      
        Zp5 = tf.sigmoid(tf.abs(s1)*(Tt-q15))       
        Yt1 = Normal(loc=(-c0[0]*Zp1+h0[0])*Maskt[0,:,:],scale = sigma_y0[0])
        Yt2 = Normal(loc=(c0[1]*Zp2+h0[1])*Maskt[1,:,:],scale = sigma_y0[1])
        Yt5 = Normal(loc=(c0[4]*Zp5+h0[4])*Maskt[2,:,:],scale = sigma_y0[4])                                                           
        data = {Xt:X_test, Tt:T_test[:,0:Me], Maskt:Mask_test[:,:,0:Me], Yt1:Y1_test[:,0:Me], 
                Yt2:Y2_test[:,0:Me], Yt5: Y5_test[:,0:Me]}
        inference = ed.KLqp({s1:q_s1, q11:q_q11, q12:q_q22, q15:q_q55},data) 
        inference.initialize()
      # tf.global_variables_initializer().run(n_iter=50000)
        inference.run(n_iter=10000)  
        j = 0

    years = 30
    d = X_test.shape[1]
    T_pre = np.arange(0,years,dtype=np.float32)/time_scale
    T_pre = np.tile(T_pre,(Ne,1))
    T_pre = T_pre+np.reshape(T_test[:,0],(Ne,1))
    
    if Me == 0:
        
        zt11 = tf.sigmoid((T_pre[j,:]-sigma_q0[0])*sigma_s0[0])
        std_mmse = -zt11.eval()*c0[0]+h0[0]
        zt22 = tf.sigmoid((T_pre[j,:]-sigma_q0[1])*sigma_s0[1])
        std_adas = zt22.eval()*c0[1]+h0[1]
        zt55 = tf.sigmoid((T_pre[j,:]-sigma_q0[4])*sigma_s0[4])
        std_cdr = zt55.eval()*c0[4]+h0[4]
        std_mmse = mmse_scale*np.reshape(std_mmse,(years,1))
        std_adas = adas_scale*np.reshape(std_adas,(years,1))
        std_cdr = cd_scale*np.reshape(std_cdr,(years,1))
    
        X_t0 =  tf.cast(X_test[j,:],tf.float32)
        X_t0 = tf.reshape(X_t0,(1,d))
        q11 = tf.matmul(X_t0,v0) + a0[0]
        q22 = tf.matmul(X_t0,v0) + a0[1]
        q55 = tf.matmul(X_t0,v0) + a0[4]
        s1 = tf.matmul(X_t0,w0) + b0
        s2 = tf.matmul(X_t0,w0) + b0
        s5 = tf.matmul(X_t0,w0) + b0
        zt1 = tf.sigmoid((T_pre[j,:]-q11)*s1)
        mean_mmse = -c0[0]*zt1.eval()+h0[0]
        zt2 = tf.sigmoid((T_pre[j,:]-q22)*s2) 
        mean_adas = c0[1]*zt2.eval()+h0[1]
        zt5 = tf.sigmoid((T_pre[j,:]-q55)*s5) 
        mean_cdr = c0[4]*zt5.eval()+h0[4]
        mean_mmse = mmse_scale*np.reshape(mean_mmse,(years,1))
        mean_adas = adas_scale*np.reshape(mean_adas,(years,1))
        mean_cdr = cd_scale*np.reshape(mean_cdr,(years,1))
       
    else:
       
        q_q1_sample = q_q11.sample(100).eval()
        q_q2_sample = q_q22.sample(100).eval()
        q_q5_sample = q_q55.sample(100).eval()
        q_s1_sample = tf.abs(q_s1.sample(100).eval())
        tmp1 = np.zeros((100,years))
        tmp2 = np.zeros((100,years))
        tmp5 = np.zeros((100,years))
       
    
        for i in range(0,100):
            zhat1 =(tf.sigmoid((T_pre[j,:]-q_q1_sample[i,j])*q_s1_sample[i,j]))
            tmp1[i,:] = mmse_scale*np.array(-c0[0]*zhat1.eval()+h0[0])
            zhat2 = (tf.sigmoid((T_pre[j,:]-q_q2_sample[i,j])*q_s1_sample[i,j]))
            tmp2[i,:] = adas_scale*np.array(c0[1]*zhat2.eval()+h0[1])
            zhat5 = (tf.sigmoid((T_pre[j,:]-q_q5_sample[i,j])*q_s1_sample[i,j]))
            tmp5[i,:] = cd_scale*np.array(c0[4]*zhat5.eval()+h0[4])
    
      #  zhat1 = (tf.sigmoid((T_pre[j,:]-q_q1_sample[:,j])*q_s1_sample[:,j]))
      #  tmp1 = mmse_scale*np.array(-c0[0]*zhat1.eval()+h0[0])
      #  zhat2 = (tf.sigmoid((T_pre[j,:]-q_q2_sample[:,j])*q_s1_sample[:,j]))
     #   tmp2 = adas_scale*np.array(c0[1]*zhat2.eval()+h0[1])
     #   zhat5 = (tf.sigmoid((T_pre[j,:]-q_q5_sample[:,j])*q_s1_sample[:,j]))
     #   tmp5 = cd_scale*np.array(c0[4]*zhat5.eval()+h0[4])
        
        mean_mmse = np.mean(tmp1[:,:],axis=0)
        std_mmse = np.std(tmp1[:,:],axis=0)
        mean_adas = np.mean(tmp2[:,:],axis=0)
        std_adas = np.std(tmp2[:,:],axis=0)
        mean_cdr = np.mean(tmp5[:,:],axis=0)
        std_cdr = np.std(tmp5[:,:],axis=0)
        
        return T_test,T_pre,std_mmse,mean_mmse,std_adas,mean_adas,std_cdr,mean_cdr,time_scale,base_age,Me,Y1_test,Y2_test,Y5_test,mmse_scale,adas_scale,cd_scale
        
