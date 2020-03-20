# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import binary_layer as binary2
from itertools import izip_longest

def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
    #return tf.truncated_normal(shape,stddev=2*d)
  return _initializer

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round_through(hard_sigmoid(x))-1.


class VAE(object):
  """ Beta Variational Auto Encoder. """
  
  def __init__(self,
               gamma=2.35,
               capacity_limit=25.0,
               capacity_change_duration=100000,
               learning_rate=1e-4,n3=32,k3=16,r3=16,EbN0dB=-2,EbN0dBb=10):

    self.learning_rate = learning_rate
    # with tf.variable_scope("gen", reuse=reuse) as scope:

 

    self.n3=n3
    self.k3=k3
    self.r3 = r3
    self.rate=self.k3/self.n3
    self.EbN0 = 10**(0.1*EbN0dB)
    self.N0 = 1/(self.EbN0*self.rate)
    self.EbN0b = 10**(0.1*EbN0dBb)
    self.N0b = 1/(self.EbN0b*self.rate)
    
    self.x_M = tf.placeholder(tf.float32, shape = (None, self.k3))
    self.x_R = tf.placeholder(tf.float32, shape = (None, self.r3))
    self.PermutationIndices=tf.placeholder(tf.int32, (None,))
    self.n_1 = tf.placeholder(tf.float32, shape = (None, self.n3))
    self.n_2 = tf.placeholder(tf.float32, shape = (None, self.n3))
    self.c_sum = tf.placeholder(tf.float32, shape = (None, self.n3))
    self.c_R_data = tf.placeholder(tf.float32, shape = (None, self.n3))   
    self.c_M_data = tf.placeholder(tf.float32, shape = (None, self.n3))
 

    # Create autoencoder network
    #self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

 

  def setLearningRate(self):
    self.learning_rate = self.learning_rate/2

  def _fc_weight_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias
  
  
  def MINE(self, XJointOrMarg, reusee):
      with tf.variable_scope("MINE1", reuse = reusee):
        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3+self.k3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")      
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,1], "fcm4")
        h_fc1 = tf.nn.relu(tf.matmul(XJointOrMarg, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc4 = tf.matmul(h_fc3b,        self.W_fcm4) + self.b_fcm4

      return h_fc4
  def MINEk(self, XJointOrMarg, reusee):
      with tf.variable_scope("MINEk1", reuse = reusee):
        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3+self.k3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")      
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,1], "fcm4")
        h_fc1 = tf.nn.relu(tf.matmul(XJointOrMarg, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc4 = tf.matmul(h_fc3b,        self.W_fcm4) + self.b_fcm4

      return h_fc4
  def MINEr(self, XJointOrMarg, reusee):
      with tf.variable_scope("MINEr1", reuse = reusee):
        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3+self.r3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")  
        self.W_fcm3c, self.b_fcm3c     = self._fc_weight_variable([500,500], "fcm3c")
        self.W_fcm3d, self.b_fcm3d     = self._fc_weight_variable([500,100], "fcm3d")       
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([100,1], "fcm4")
        h_fc1 = tf.nn.relu(tf.matmul(XJointOrMarg, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc3c = tf.nn.relu(tf.matmul(h_fc3b,        self.W_fcm3c) + self.b_fcm3c)
        h_fc3d = tf.nn.relu(tf.matmul(h_fc3c,        self.W_fcm3d) + self.b_fcm3d)
        h_fc4 = tf.matmul(h_fc3d,        self.W_fcm4) + self.b_fcm4

      return h_fc4      
  def binary2_activation(self,x):

        cond = tf.less(x, 0.5*tf.ones(tf.shape(x)))
        out = tf.where(cond, -tf.ones(tf.shape(x)), tf.ones(tf.shape(x)))

        return out





  def H_encoder(self, x1,x2, reusee):
      with tf.variable_scope('H_encoder', reuse = reusee):

       
        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.k3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
           
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,256], "fcm4")

        self.W_fcm1W, self.b_fcm1W     = self._fc_weight_variable([self.r3, 500], "fcm1W")
        self.W_fcm2W, self.b_fcm2W     = self._fc_weight_variable([500, 500], "fcm2W")
        self.W_fcm3W, self.b_fcm3W     = self._fc_weight_variable([500,500], "fcm3W")
            
        self.W_fcm4W, self.b_fcm4W     = self._fc_weight_variable([500,256], "fcm4W")



        self.W_fcm9W, self.b_fcm9W     = self._fc_weight_variable([1024,self.n3], "fcm9W")


        jjk=x1#tf.concat([x1,x2],1)
        h_fc1 = tf.nn.relu(tf.matmul(jjk, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
      
        h_fc4 = tf.nn.tanh(tf.matmul(h_fc3,        self.W_fcm4) + self.b_fcm4)
  

        jjkW=x2
        h_fc1W = tf.nn.relu(tf.matmul(jjkW, self.W_fcm1W) + self.b_fcm1W)
        h_fc2W = tf.nn.relu(tf.matmul(h_fc1W,        self.W_fcm2W) + self.b_fcm2W)
        h_fc3W = tf.nn.relu(tf.matmul(h_fc2W,        self.W_fcm3W) + self.b_fcm3W)
       
        h_fc4W = tf.nn.tanh(tf.matmul(h_fc3W,        self.W_fcm4W) + self.b_fcm4W)
   
        h_fc4V=h_fc4 -40* h_fc4W

        input_layer = tf.reshape(h_fc4V, [-1, 32, 8])
        conv1 = tf.layers.conv1d(inputs=input_layer, filters=16, kernel_size=4, padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)

        cnn = tf.layers.flatten(conv2)

        h_fc6 =  binary_tanh_unit(tf.matmul(cnn,        self.W_fcm9W) + self.b_fcm9W) 

      return h_fc6

  def B_decoder(self, x, reusee):
      with tf.variable_scope('B_decoder', reuse = reusee):


        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")     
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,self.k3], "fcm4")
        self.W_fcm4X, self.b_fcm4X     = self._fc_weight_variable([500,self.r3], "fcm4X")



        h_fc1 = tf.nn.relu(tf.matmul(x, self.W_fcm1) + self.b_fcm1)


        h_fc2 = tf.nn.relu(tf.matmul(h_fc1 ,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3b,        self.W_fcm4) + self.b_fcm4)
        h_fc4X = tf.nn.sigmoid(tf.matmul(h_fc3b,        self.W_fcm4X) + self.b_fcm4X)

      return h_fc4,h_fc4X



  def B_decoder_for_random(self, x, reusee):
      with tf.variable_scope('B_decoder_for_random', reuse = reusee):

   
        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")      
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,self.r3], "fcm4")
        h_fc1 = tf.nn.relu(tf.matmul(x, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3b,        self.W_fcm4) + self.b_fcm4)
        #h_fc5 = self.fully_connect_bn(h_fc4, self.r3, act=binary2.binary_tanh_unit, use_bias=True, training=True,name='B_decoder_Random5')
      return h_fc4#self.kossher(h_fc3jj)#self.binary2_activation(h_fc3jj)


  def E_decoder(self, x, reusee):
      with tf.variable_scope('E_decoder', reuse = reusee):

        self.W_fcm1, self.b_fcm1     = self._fc_weight_variable([self.n3, 500], "fcm1")
        self.W_fcm2, self.b_fcm2     = self._fc_weight_variable([500, 500], "fcm2")
        self.W_fcm3, self.b_fcm3     = self._fc_weight_variable([500,500], "fcm3")
        self.W_fcm3b, self.b_fcm3b     = self._fc_weight_variable([500,500], "fcm3b")    
        self.W_fcm4, self.b_fcm4     = self._fc_weight_variable([500,self.k3], "fcm4")




        h_fc1 = tf.nn.relu(tf.matmul(x, self.W_fcm1) + self.b_fcm1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1 ,        self.W_fcm2) + self.b_fcm2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,        self.W_fcm3) + self.b_fcm3)
        h_fc3b = tf.nn.relu(tf.matmul(h_fc3,        self.W_fcm3b) + self.b_fcm3b)
        h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3b,        self.W_fcm4) + self.b_fcm4)

      return h_fc4
  


 
  def _create_loss_optimizer(self):
    # Reconstruction loss

    self.lr1 = tf.placeholder(tf.float32, shape=[])

    
 
 
    


        # encoder, labelled data
    self.c_M= self.H_encoder(self.x_M,self.x_R,  False)

    self.c3 =self.c_M# self.Sum_I_Guess(self.c_R,self.c_M,False)#self.c_R#self.Sum_I_Guess(self.c_R,self.c_M,reusee = False)#self.kossher20(self.c_R+self.c_M)#(-1)**self.kossher2(self.c_R+self.c_M)#self.mod2_activation(self.c_R+self.c_M)
    self.y3=self.c3+np.sqrt(self.N0/2)*self.n_1#np.random.normal(0,1,[100,self.n3])
    self.y3b=self.c3+np.sqrt(self.N0b/2)*self.n_2#np.random.normal(0,1,[100,self.n3])

   
      decodedM,decodedR=self.B_decoder(self.y3b,  reusee = False)
    self.ML1=tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_M-decodedM), axis=1))
    self.ML1_for_random=tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_R-decodedR), axis=1))#tf.abs(0.05-tf.reduce_mean(tf.reduce_sum(tf.abs((-1)**self.x_M-self.B_decoder(self.y3b,  reusee = False)), axis=1))/10)


    EdecodedM=self.E_decoder(self.y3,  reusee = False)
    self.Eveloss =  tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_M-EdecodedM), axis=1))#tf.reduce_mean(tf.reduce_sum(tf.abs((-1)**self.x_M-self.E_decoder(self.y3,  reusee = False)), axis=1))
    
    E_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='E_decoder')
    self.EoptimizerRandom  = tf.train.AdamOptimizer(self.lr1).minimize(self.Eveloss  , var_list=E_decoder_vars)


    self.BobBer=tf.reduce_mean(tf.to_float(tf.not_equal(2*self.x_M-1,self.binary2_activation(decodedM))))
    self.BobBer_for_random=tf.reduce_mean(tf.to_float(tf.not_equal(2*self.x_R-1,self.binary2_activation(decodedR))))
    self.EveBer=self.BobBer



    self.xMINE=self.y3
    self.yMINE= self.x_M
    self.XJoint=tf.concat([self.xMINE, self.yMINE], 1)
    self.xMINEshuff=tf.gather(self.xMINE, self.PermutationIndices)
    self.XMarg=tf.concat([self.xMINEshuff, self.yMINE], 1)
    y2_joint= self.MINE(self.XJoint,reusee = False)  
    y2_marg= self.MINE(self.XMarg,reusee = True) 
    aya2z=tf.reduce_mean(tf.exp(y2_marg))
    aya22z=tf.reduce_mean(y2_joint)
    MI=aya22z-tf.log(aya2z)
    self.MINEloss=-MI



    self.xMINE=self.c3
    self.yMINE= self.x_M
    self.XJoint=tf.concat([self.xMINE, self.yMINE], 1)
    self.xMINEshuff=tf.gather(self.xMINE, self.PermutationIndices)
    self.XMarg=tf.concat([self.xMINEshuff, self.yMINE], 1)
    y2_joint= self.MINEk(self.XJoint,reusee = False)  
    y2_marg= self.MINEk(self.XMarg,reusee = True) 
    aya2z=tf.reduce_mean(tf.exp(y2_marg))
    aya22z=tf.reduce_mean(y2_joint)
    MI=aya22z-tf.log(aya2z)
    self.MINElossk=-MI



    self.xMINE=self.c3
    self.yMINE= self.x_R
    self.XJoint=tf.concat([self.xMINE, self.yMINE], 1)
    self.xMINEshuff=tf.gather(self.xMINE, self.PermutationIndices)
    self.XMarg=tf.concat([self.xMINEshuff, self.yMINE], 1)
    y2_joint= self.MINEr(self.XJoint,reusee = False)  
    y2_marg= self.MINEr(self.XMarg,reusee = True) 
    aya2z=tf.reduce_mean(tf.exp(y2_marg))
    aya22z=tf.reduce_mean(y2_joint)
    MI=aya22z-tf.log(aya2z)
    self.MINElossr=-MI


    MINE1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MINE1')
    MINEk1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MINEk1')
    MINEr1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MINEr1')


    self.optimizerMINE1 = tf.train.AdamOptimizer(learning_rate=self.lr1).minimize(self.MINEloss, var_list=MINE1_vars)
    self.optimizerMINEk1 = tf.train.AdamOptimizer(learning_rate=self.lr1).minimize(self.MINElossk, var_list=MINEk1_vars)
    self.optimizerMINEr1 = tf.train.AdamOptimizer(learning_rate=self.lr1/2).minimize(self.MINElossr, var_list=MINEr1_vars)

    self.ML2=-self.MINEloss#(tf.abs(MI-0.05))#MI#
    self.Mainloss = self.ML1+self.ML1_for_random
    self.MainlossB =self.ML1_for_random
    self.Mainloss2 =(tf.abs(self.ML2))
    


   


    def JJDOUBLE(xx1,xx2):
        if (xx1 is None) and (xx2 is None):
            return None
        elif (xx1 is None) and (xx2 is not None):
            return xx2
        elif (xx1 is not None) and (xx2 is  None):
            return  None
        elif (xx1 is not None) and (xx2 is not  None):
            return tf.minimum(tf.norm(xx1),tf.norm(xx2))*(xx1/tf.norm(xx1))#xx2+0.25*xx1#




    optT = binary2.AdamOptimizer(binary2.get_all_LR_scale(), self.lr1)


    optTB = binary2.AdamOptimizer(binary2.get_all_LR_scale(), self.lr1)
 
    opt2TB = tf.train.AdamOptimizer(self.lr1)
    


    dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='H_encoder')
    B_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='B_decoder')

    B_decoder_vars_for_random = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='B_decoder_for_random')




    optA = binary2.AdamOptimizer(binary2.get_all_LR_scale(), self.lr1)
    optA2 = tf.train.AdamOptimizer(self.lr1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    optimizerMessageVARS=[var for var in dec_vars ]
    optimizerMessageVARSBinary= [var for var in dec_vars]
    



    gvsBERMessage = opt2TB.compute_gradients(self.Mainloss, var_list=optimizerMessageVARS+B_decoder_vars)
    gvsMIk = opt2TB.compute_gradients(self.MINElossk+self.Mainloss2, var_list=optimizerMessageVARS)



    gvsTEST = opt2TB.compute_gradients(self.Mainloss, var_list=optimizerMessageVARS )
    gradTEST,varTEST=zip(*gvsTEST )
    self.wwgg=gradTEST

    gradgvsBER,vargvsBER=zip(*gvsBERMessage )
    gradgvsMIk,vargvsMIk=zip(*gvsMIk)
    #gradgvsMIr,vargvsMIr=zip(*gvsMIr)
    gradgvsMIk=gradgvsMIk+ tuple([None] * (len(gradgvsBER)-len(gradgvsMIk)))

    assert len(gradgvsBER)==len(gradgvsMIk)#==len(gradgvsMIr)
    wholeGradJJ=[JJDOUBLE(grad1,grad2) for grad1, grad2 in zip(gradgvsMIk, gradgvsBER)]
    #print(wholeGradJJ)
    self.optimizerMessage  = opt2TB.apply_gradients(zip(wholeGradJJ,vargvsBER))
  

        



  def partial_fit2(self, sess, x_M,x_R,lr2,n_1,n_2):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
                                       

  
                                             
    k4=range(0,len(x_M))#                                                               
    np.random.shuffle(k4)                
    _,_,BobBer,EveBer,cc,ml1,ml2,cm,berrandom,wg = sess.run((self.optimizerMessage,self.EoptimizerRandom,self.BobBer,self.EveBer,self.c3,self.ML1,self.ML2,self.c_M,self.BobBer_for_random,self.wwgg),
                                    feed_dict={                   
                                    self.x_M : x_M,self.n_1:n_1,
                                    self.x_R:x_R,self.n_2:n_2,
                                    self.lr1:lr2,
                                    self.PermutationIndices:k4
                                    })

				 
    return BobBer,EveBer,cc,ml1,ml2,cm,cm,berrandom ,wg

  def partial_fit_MINE222(self, sess, x_M,x_R,lr2,n_1,n_2):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """

    k4=range(0,len(x_M))#
    np.random.shuffle(k4)                                  #self.Eveoptimizer00,self.Eveoptimizer,

    _,_,Mine,minek= sess.run((self.optimizerMINE1,self.optimizerMINEk1,self.MINEloss,self.MINElossk),
                                                  feed_dict={
                                                            self.x_M : x_M,self.n_1:n_1,
                                                            self.x_R:x_R,self.n_2:n_2,
                                                            self.lr1:lr2,self.PermutationIndices:k4
                                                 })
    miner  =minek                                               				 
    return -Mine,-minek,-miner

