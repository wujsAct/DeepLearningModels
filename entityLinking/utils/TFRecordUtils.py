# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''

import os
import tensorflow as tf
import numpy as np
import cPickle
import time

'''
@author:wujs
function: å°pickleçç¹å¾è½¬æ¢æTFRecordæ¹ä¾¿tensorflowè®­ç»æ¶è°ï¿½ï¿½?Named Entity Recognition parameter:
 [1]:featureArray shape [batch_size,sentence_length,feature_dims]
 [2]:labelArray shape [batch_size,sentence_length,tag_dim]
'''
def ner_d3array_TFRecord(featureArray,labelArray,TFfileName,nerShapeFile):
  writer = tf.python_io.TFRecordWriter(TFfileName)
  fShape = np.shape(featureArray)
  lShape = np.shape(labelArray)
  print 'feature shape:',fShape
  print 'label shape:',lShape
  assert fShape[0]==lShape[0]
  assert fShape[1]==lShape[1]
  ldim = lShape[1]*lShape[2]; fdim = fShape[1]*fShape[2];
  for i in xrange(fShape[0]):
    lab = np.reshape(labelArray[i],[ldim])
    feat = np.reshape(featureArray[i],[fdim])
    example = tf.train.Example(features=tf.train.Features(feature={
     "label":tf.train.Feature(int64_list=tf.train.Int64List(value=lab)),
     "feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat))
    }))
    writer.write(example.SerializeToString())
  writer.close()
  param_dict={'fShape':fShape,'lShape':lShape}
  cPickle.dump(param_dict,open(nerShapeFile,'wb'))

def ner_d3array_read_and_decode(TFfileName,nerShapeFile,num_epochs):
  param_dict= cPickle.load(open(nerShapeFile))
  fShape = param_dict['fShape'];lShape = param_dict['lShape']
  ldim = lShape[1]*lShape[2]; fdim = fShape[1]*fShape[2];
  fileName_queue=tf.train.string_input_producer([TFfileName],num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(fileName_queue)
  features = tf.parse_single_example(serialized_example,
                                     features={'label':tf.FixedLenFeature([ldim],tf.int64),
                                               'feature':tf.FixedLenFeature([fdim],tf.float32)})
  label = tf.cast(features['label'],tf.int32)
  feature = tf.cast(features['feature'],tf.float32)
  return feature,label


def ner_read_TFRecord(sess,TFfileName,nerShapeFile,batch_size,num_epochs):
  param_dict= cPickle.load(open(nerShapeFile))
  fShape = param_dict['fShape'];lShape = param_dict['lShape']
  feature,label = ner_d3array_read_and_decode(TFfileName,nerShapeFile,num_epochs)
  sess.run(tf.local_variables_initializer())
  feature_batch,label_batch = tf.train.shuffle_batch([feature,label],
                                                     batch_size=batch_size,
                                                     num_threads=4,capacity=50000,
                                                     min_after_dequeue=10000)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  
  try:
    while not coord.should_stop():
      val,l = sess.run([feature_batch,label_batch])
      yield np.reshape(val,[batch_size,fShape[1],fShape[2]]),np.reshape(l,[batch_size,lShape[1],lShape[2]])
  except tf.errors.OutOfRangeError:
    print 'Done training -- epoch limit reached'
  finally:
    #When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)


'''
param_dict={'ent_mention_linking_tag':[lents,args.candidate_ent_num],
              'candidate_ent_linking_feature':[lents,args.candidate_ent_num,int(args.rawword_dim)],
              'candidate_ent_coherent_feature':[lents,args.candidate_ent_num],
              'candidate_ent_type_feature':[lents,args.candidate_ent_num,args.figer_type_num],
              'candidate_ent_prob_feature':[lents,args.candidate_ent_num,3],
              'ent_mention_lstm_feature':[lents,2*args.rnn_size,1]
              }  
'''
def nel_d3array_read_and_decode(TFfileName,nerShapeFile,num_epochs):
  param_dict = cPickle.load(open(nerShapeFile))
  print param_dict
  label_shape = param_dict['ent_mention_linking_tag']
  feat1_shape = param_dict['candidate_ent_coherent_feature']
  feat2_shape = param_dict['candidate_ent_linking_feature']
  feat3_shape = param_dict['candidate_ent_type_feature']
  feat4_shape = param_dict['candidate_ent_prob_feature']
  feat5_shape = param_dict['ent_mention_lstm_feature']
  
  ldim = label_shape[1];
  f1dim = feat1_shape[1]; f2dim = feat2_shape[1] * feat2_shape[2];
  f3dim = feat3_shape[1] * feat3_shape[2];f4dim = feat4_shape[1] * feat4_shape[2];
  f5dim = feat5_shape[1] * feat5_shape[2];
  
  fileName_queue=tf.train.string_input_producer([TFfileName],num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(fileName_queue)
  
  features = tf.parse_single_example(serialized_example,
                                     features={'ent_mention_linking_tag':tf.FixedLenFeature([ldim],tf.int64),
                                               'candidate_ent_coherent_feature':tf.FixedLenFeature([f1dim],tf.float32),
                                               'candidate_ent_linking_feature':tf.FixedLenFeature([f2dim],tf.float32),
                                               'candidate_ent_type_feature':tf.FixedLenFeature([f3dim],tf.float32),
                                               'candidate_ent_prob_feature':tf.FixedLenFeature([f4dim],tf.float32),
                                               'ent_mention_lstm_feature':tf.FixedLenFeature([f5dim],tf.float32)
                                               })
  label = tf.cast(features['ent_mention_linking_tag'],tf.float32)
  f1 = features['candidate_ent_coherent_feature']
  f2 = features['candidate_ent_linking_feature']
  f3= features['candidate_ent_type_feature']
  f4 = features['candidate_ent_prob_feature']
  f5 = features['ent_mention_lstm_feature']
  return f1,f2,f3,f4,f5,label
  
def nel_read_TFRecord(sess,TFfileName,nerShapeFile,batch_size,num_epochs):
  '''
  param_dict={'ent_mention_linking_tag':[lents,args.candidate_ent_num],
              'candidate_ent_coherent_feature':[lents,args.candidate_ent_num],
              'candidate_ent_linking_feature':[lents,args.candidate_ent_num,int(args.rawword_dim)],
              'candidate_ent_type_feature':[lents,args.candidate_ent_num,args.figer_type_num],
              'candidate_ent_prob_feature':[lents,args.candidate_ent_num,3],
              'ent_mention_lstm_feature':[lents,2*args.rnn_size,1]
              }
  '''
  param_dict = cPickle.load(open(nerShapeFile))
  feat2_shape = param_dict['candidate_ent_linking_feature']
  feat3_shape = param_dict['candidate_ent_type_feature']
  feat4_shape = param_dict['candidate_ent_prob_feature']
  feat5_shape = param_dict['ent_mention_lstm_feature']
  
  f1,f2,f3,f4,f5,label = nel_d3array_read_and_decode(TFfileName,nerShapeFile,num_epochs)
  
  f1_batch,f2_batch,f3_batch,f4_batch,f5_batch,label_batch = tf.train.batch([f1,f2,f3,f4,f5,label],
                                                     batch_size=batch_size)
                                                     
  
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  try:
    while not coord.should_stop():
      f1,f2,f3,f4,f5,l = sess.run([f1_batch,f2_batch,f3_batch,f4_batch,f5_batch,label_batch])
      #yield l,f1,f2,f3,f4,f5
      yield l,f1,np.reshape(f2,[batch_size,feat2_shape[1],feat2_shape[2]]),\
            np.reshape(f3,[batch_size,feat3_shape[1],feat3_shape[2]]),np.reshape(f4,[batch_size,feat4_shape[1],feat4_shape[2]]),\
            np.reshape(f5,[batch_size,feat5_shape[1],feat5_shape[2]])
  except tf.errors.OutOfRangeError:
    print 'Done training -- epoch limit reached'
  finally:
    #When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
  

if __name__=="__main__":
  nerfile = "ner.tfrecords"
  nershape = "ner.shape"
  #mu = 0;sigma=1
  #rarray=np.random.normal(mu,sigma,[2580,2,5])
  #label=np.zeros([2580,2,3],dtype=np.int64)
  #ner_d3array_TFRecord(rarray,label,nerfile,nershape)
  TFfileName = '../data/aida/trainNEL.tfrecord'
  nerShapeFile ='../data/aida/trainNEL.shape'
  
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  sess.run(tf.global_variables_initializer())
  for l,f1,f2,f3,f4,f5 in nel_read_TFRecord(sess,TFfileName,nerShapeFile,32,2):
    print np.shape(l),np.shape(f1),np.shape(f2),np.shape(f3),np.shape(f4),np.shape(f5)
  sess.close()
