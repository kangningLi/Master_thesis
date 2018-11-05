#!/usr/bin/python3



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import tensorflow as tf
import numpy as np
import pointfly as pf
BASE_DIR=os.path.dirname(os.path.abspath(__file__))   ##/home/kangning/Documents/Masterarbeit/frus_exp/models
sys.path.append(os.path.join(BASE_DIR,'sampling/'))


def xconv(pts,fts,qrs,tag,N,K,D,P,C,depth_multiplier,is_training,C_pts_fts,with_x_transformation=True,sorting_method=None,with_global=False):
###xconv(pts,fts,qrs,tag,N,K,D,P,C,C_pts_fts,is_training,with_X_transformation,depth_multipiler,sorting_method,with_global)

  # @params:
  #     N: Number of output points
  #     K: Number of nearest neighbot
  #     D: dilation rate
  #     C: Number of output channels
  #     P:the representative point number in the output, -1 means all input points are output representative points
  #     x_transformation: replace max_pooling in PointNet
  #     sorting_method:
  #     with_global
  #     pts:Input point cloud
  #     qrs:queries
  #     fts:features

    _, indices_dilated= pf.knn_indices_general(qrs,pts,K*D,True)


    indices= indices_dilated[:,:,::D,:]
    if sorting_method is not None:
        indices=pf.sort_points(pts,indices,sorting_method)

    nn_pts= tf.gather_nd(pts,indices,name=tag +'nn_pts') #(N,P,K,3)
    nn_pts_center=tf.expand_dims(qrs,axis=2, name= tag+'nn_pts_center') #(N,P,1,3)
    nn_pts_local=tf.subtract(nn_pts,nn_pts_center,name= tag+'nn_pts_local') #(N,P,K,3)


    #Prepare features to be transformed


    nn_fts_from_pts_0=pf.dense(nn_pts_local, C_pts_fts, tag+ 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts=pf.dense(nn_fts_from_pts_0,C_pts_fts,tag+ 'nn_fts_from_pts', is_training)
    
    if fts is None:
        nn_fts_input=nn_fts_from_pts
    else:
        nn_fts_from_prev= tf.gather_nd(fts,indices,name=tag+'nn_fts_from_prev')
        nn_fts_input=tf.concat([nn_fts_from_pts,nn_fts_from_prev],axis=-1,name=tag+ 'nn_fts_input')


    #X_transformation
    if with_x_transformation:
        ############################X_transformation#######################################################

        X_0=pf.conv2d(nn_pts_local,K*K,tag+'X_0',is_training,(1,K))
        X_0_KK=tf.reshape(X_0,(N,P,K,K),name=tag+'X_0_KK')

        X_1=pf.depthwise_conv2d(X_0_KK,K,tag+'X_1',is_training,(1,K))
        X_1_KK=tf.reshape(X_1,(N,P,K,K),name=tag+'X_1_KK')

        X_2=pf.depthwise_conv2d(X_1_KK,K,tag+'X_2',is_training,(1,K),activation=None)
        X_2_KK=tf.reshape(X_2,(N,P,K,K),name=tag+'X_2_KK')
        fts_X=tf.matmul(X_2_KK,nn_fts_input,name=tag+'fts_X')
    
        #####################################################################################################

    else:
        fts_X=nn_fts_input

    fts_conv=pf.separable_conv2d(fts_X,C,tag+ 'fts_conv',is_training,(1,K),depth_multiplier=depth_multiplier)
    fts_conv_3d=tf.squeeze(fts_conv,axis=2,name=tag+'fts_conv_3d')

    if with_global:
        fts_global_0=pf.dense(qrs,C//4,tag+'fts_global_0',is_training)
        fts_global=pf.dense(fts_global_0,C//4,tag+'fts_global',is_training)
        return tf.concat([fts_global,fts_conv_3d],axis=-1,name=tag+'fts_conv_3d_with_global')

    else:
        return fts_conv_3d




def Invariance_Transformation_Net(point_cloud,features, is_training, invarians_trans_param):
    xconv_params=invarians_trans_param.xconv_params
    fc_params=invarians_trans_param.fc_params
    with_X_transformation=invarians_trans_param.with_X_transformation
    sorting_method=invarians_trans_param.sorting_method
    #N=point_cloud.get_shape()[1].value
    N=point_cloud.get_shape()[0].value
    if invarians_trans_param.sampling=='fps':
        import tf_sampling

    layer_pts=[point_cloud]
    if features is None:
        layer_fts=[features]
    else:
        features=tf.reshape(features,(N,-1,invarians_trans_param.data_dim-3),name='features_reshape')
        C_fts=xconv_params[0]['C']//2
        features_hd=pf.dense(features,C_fts,'features_hd',is_training)
        layer_fts=[features_hd]
    for layer_idx, layer_param in enumerate(xconv_params):
        tag = 'xconv_' +str(layer_idx+1)+'_' #####xconv_1_ #####xconv_2_ #####xconv_3_ #####xconv_4_
        K=layer_param['K']
        D=layer_param['D']
        P=layer_param['P']
        C=layer_param['C']
        links=layer_param['links']   ## type(layer_param) is dict

        #get k-nearest points
        pts=layer_pts[-1]
        fts=layer_fts[-1]
        if P== -1 or (layer_idx>0 and P==xconv_params[layer_idx -1]['P']):
            qrs=layer_pts[-1]
        else:
            if invarians_trans_param.sampling=='fps':
                fps_indices= tf_sampling.farthest_point_sample(P,pts)
                batch_indices= tf.tile(tf.reshape(tf.range(N),(-1,1,1)),(1,P,1))
                indices =tf.concat([batch_indices,tf.expand_dims(fps_indices,-1)],axis=-1)
                qrs=tf.gather_nd(pts,indices,name=tag+'qrs') ### (N,P,3)
                print(tf.shape(qrs))

            elif invarians_trans_param.sampling=='ids':
                indices=pf.inverse_density_sampling(pts,K,P)
                qrs= tf.gather_nd(pts,indices)

            elif invarians_trans_param=='random':
                qrs= tf.slice(pts,(0,0,0),(-1,P,-1),name=tag+'qrs')### (N,P,3)

            else:
                print('unknown sampling method')
                exit()
        layer_pts.append(qrs)

        
        if layer_idx==0:
            C_pts_fts=C//2 if fts is None else C//4
            depth_multipiler=4

        else:
            C_prev=xconv_params[layer_idx-1]['C']
            C_pts_fts=C_prev // 4
            depth_multipiler=math.ceil(C/C_prev)
        with_global=(invarians_trans_param.with_global and layer_idx ==len(xconv_params)-1)
  
        fts_xconv=xconv(pts,fts,qrs,tag,N,K,D,P,C,C_pts_fts,is_training,with_X_transformation,depth_multipiler,sorting_method,with_global)
        
        fts_list=[]
        for link in links:
            fts_from_link =layer_fts[link]
            if fts_from_link is not None:
                fts_slice=tf.slice(fts_from_link,(0,0,0),(-1,P,-1),name=tag+'fts_slice'+str(-link))
                fts_list.append(fts_slice)

        if fts_list:
            fts_list.append(fts_xconv)
            layer_fts.append(tf.concat(fts_list,axis=-1,name=tag+'fts_list_concat'))
   
        else:
            layer_fts.append(fts_xconv)


    if hasattr(invarians_trans_param,'xdconv_params'):
        for layer_idx, layer_param in enumerate(invarians_trans_param.xdconv_params):
            tag='xdconv_' +str(layer_idx +1)+'_'
            K=layer_param['K']
            D=layer_param['D']
            pts_layer_idx=layer_param['pts_layer_idx']
            qrs_layer_idx=layer_param['qrs_layer_idx']


            pts=layer_pts[pts_layer_idx+1]
            fts=layer_fts[pts_layer_idx+1] if layer_idx==0 else layer_fts[-1]
            qrs=layer_pts[qrs_layer_idx+1]
            fts_qrs=layer_fts[qrs_layer_idx+1]
             
            P=xconv_params[qrs_layer_idx]['P']
            C=xconv_params[qrs_layer_idx]['C']
            C_prev=xconv_params[pts_layer_idx]['C']
            C_pts_fts=C_prev // 4
            depth_multipiler=1

            fts_xdconv=xconv(pts,fts,qrs,tag,N,K,D,P,C,C_pts_fts,is_training,with_X_transformation,depth_multipiler,sorting_method)	
            fts_concat=tf.concat([fts_xdconv,fts_qrs],axis=-1,name=tag+'fts_concat')
            fts_fuse=pf.dense(fts_concat,C,tag+'fts_fuse',is_training)
            layer_pts.append(qrs)
            layer_fts.append(fts_fuse)




    fc_layers= layer_fts[-1]
    for layer_idx,layer_param in enumerate(fc_params):
        C=layer_param['C']
        dropout_rate= layer_param['dropout_rate']
        fc=pf.dense(fc_layers[-1],C,'fc{:d}'.format(layer_idx),is_training)
        fc_drop=tf.layers.dropout(fc,dropout_rate,training=is_training,name='fc{:d}_drop'.format(layer_idx))
        fc_layers.append(fc_drop)
    return fc_layers
            
   



          
