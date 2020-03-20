# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
#from scipy.misc import imsave
import time
from modelbin2norandom import VAE

from sklearn.metrics import accuracy_score
import numpy.matlib


tf.app.flags.DEFINE_integer("epoch_size", 40000000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 2048, "batch size")
tf.app.flags.DEFINE_integer("Lbatch_size", 2, "Labled batch size")
tf.app.flags.DEFINE_float("gamma", 1, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 25.0,
                          "encoding capacity limit paramd for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 2.5e-5, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints-VAE-CFR", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

flags = tf.app.flags.FLAGS

def train(sess,
          model,saveba):


  





  step = 0
  rl=[]
  prr=[]
  ll=[]
	

  lera=4e-5

  print(lera)
  minees=[]
  minee2s=[]
  MIIs=[]
  MIIsk=[]
  MIIsr=[]
  randombers2=[]
  n=16
  k=5
  r=11
  randomMSGs=np.array(np.zeros((2**r,r)))
  for i in range(2**r):
      c='{0:011b}'.format(i)
      randomMSGs[i]=[int(j) for j in list(c)]
  print(randomMSGs)
  randomMSGs=np.matlib.repmat(randomMSGs, 20000, 1)
  secretMSGs=np.array(np.zeros((2**k,k)))
  for i in range(2**k):
      c='{0:05b}'.format(i)
      secretMSGs[i]=[int(j) for j in list(c)]
  secretMSGs=np.matlib.repmat(secretMSGs, 200000, 1)

  for epoch in range(flags.epoch_size):

    print(epoch)
    total_batch = 8*10

    if (epoch > 0 and epoch%400==0  and epoch<400*2+2):  
      lera=lera/2
      print('rate just got halved!')

    minee2=0
    minee=0
    mll1=0
    mll2=0
    randombers=0
    cc=[]

    MII=0
    MIIk=0
    MIIr=0		
 


    if epoch<999999999999:
      for i in range(total_batch):
        message_bits=np.random.choice([0, 1], size=(flags.batch_size,k), p=[1./2, 1./2])

        random_bits=randomMSGs[i*flags.batch_size:(i+1)*flags.batch_size]#np.matlib.repmat(randomMSGs, 1, 1)#randomMSGs

        np.random.shuffle(random_bits)


        n1=np.random.normal(0,1,[flags.batch_size,n])
        n2=np.random.normal(0,1,[flags.batch_size,n])

        
        if epoch<-20:
  
        else:
          BobBer,EveBer,cc,ml1,ml2,CM,CR,randomber,wg  = model.partial_fit2(sess,message_bits,random_bits,lera,n1,n2)
          MI,MIk,MIr  = model.partial_fit_MINE222(sess,message_bits,random_bits,lera,n1,n2)
        
          mll1=mll1+ml1
          mll2=mll2+ml2
          minee=minee+BobBer
          minee2=minee2+EveBer
          randombers=randombers+randomber
       



        MII=MII+MI
        MIIk=MIIk+MIk
        MIIr=MIIr+MIr



    step += 1

 
    if epoch>0:
      randombers2.append(randombers/total_batch)    
      minees.append(minee/total_batch)
      minee2s.append(minee2/total_batch)
      MIIs.append(MII/total_batch)
      MIIsk.append(MIIk/total_batch)
      MIIsr.append(MIIr/total_batch)
      #print('scorse : '+str(scorse))
      #print(wg)
      print(cc)
      print("ML1:",mll1/total_batch)
      print("ML2:",mll2/total_batch)
      print("MAIN CHANNEL BER:",minee/total_batch)
      print("Random BER:",randombers/total_batch)
      print("EVE's MI:",MII/total_batch)
      print("K's MI:",MIIk/total_batch)
      print("R's MI:",MIIr/total_batch)



    # Save checkpoint
    if epoch>0 and epoch%100 ==0:
      saveba.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)


def load_checkpoints(sess):
   saver = tf.train.Saver()
   checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
   if checkpoint and checkpoint.model_checkpoint_path:
     saver.restore(sess, checkpoint.model_checkpoint_path)
     print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
   else:
     print("Could not find old checkpoint")
     if not os.path.exists(flags.checkpoint_dir):
       os.mkdir(flags.checkpoint_dir)
   return saver


def main(argv):

  sess = tf.Session()
  
  model = VAE(gamma=flags.gamma,
              capacity_limit=flags.capacity_limit,
              capacity_change_duration=flags.capacity_change_duration,
              learning_rate=flags.learning_rate,n3=16,k3=5,r3=11,EbN0dB=-4,EbN0dBb=10)
  
  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

  if flags.training:
    # Train
    train(sess, model, saver)
  

if __name__ == '__main__':
  tf.app.run()
