# -*- coding: utf-8 -*-
'''
@editor: wujs
function: we add entity linking module
revise: 2017/1/8
'''

import tensorflow as tf
import time
from model import ctxSum
from sklearn.metrics import f1_score
from utils import nelInputUtilsDoc as inputUtils
from utils import getLinkingFeature
import numpy  as np
import cPickle
import argparse
from utils import genEntMentMask
import pprint

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",10,"batch size of training")
flags.DEFINE_string("datasets","ace","hidden dimension of rnn")
flags.DEFINE_integer("sentence_length",250,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",310,"hidden dimension of rnn")
flags.DEFINE_integer("ner_word_dim",300,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",50,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.005,"apply dropout during training")

args = flags.FLAGS

def getData(dataTag):
  dataUtils = inputUtils(args.rawword_dim,dataTag)
  data_batch_size = dataUtils.batch_size
  data_input = dataUtils.emb; 
  data_entliking= dataUtils.ent_linking;data_ent_mention_index = data_entliking['ent_mention_index'];
  data_ent_mention_link_feature=data_entliking['ent_mention_link_feature'];data_ent_mention_tag = data_entliking['ent_mention_tag']; 
  data_ent_relcoherent = [dataUtils.ent_relcoherent_ngd,dataUtils.ent_relcoherent_fb]
  data_ent_linking_type = dataUtils.ent_linking_type; 
  data_ent_linking_candprob = dataUtils.ent_linking_candprob
  data_ent_surfacewordv_feature = dataUtils.ent_surfacewordv_feature
  
  data_shape = np.shape(data_input)
  if dataTag in ['train','testa','testb']:
    data_out = np.argmax(dataUtils.tag,2) 
  else:
    data_out = np.zeros([data_shape[0],data_shape[1],args.class_size],dtype=np.float32)  
    data_out = np.argmax(data_out,2)
  
  return data_batch_size,data_input, data_out, data_ent_mention_index, data_ent_mention_link_feature, data_ent_mention_tag, data_ent_relcoherent,data_ent_linking_type,data_ent_linking_candprob,data_ent_surfacewordv_feature


#def getDataTestSets():
#  kbpTestList = getData("kbp_evaluation")
#  
#  kbpNums = np.shape(kbpTestList[1])[0]; kbpTestRange= int(kbpNums*0.95)
#  print kbpNums
#  valRet=[]
#  testRet=[]
#
#  for i in range(len(kbpTestList)):
#    if i==5:
#      valRet.append([kbpTestList[i][0][kbpTestRange:kbpNums],kbpTestList[i][1][kbpTestRange:kbpNums]])
#      testRet.append([kbpTestList[i][0][0:kbpTestRange],kbpTestList[i][1][0:kbpTestRange]])
#    else:
#      valRet.append(kbpTestList[i][kbpTestRange:kbpNums])
#      testRet.append(kbpTestList[i][0:kbpTestRange])
#    
#  return valRet,testRet
  
  

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--features', type=str, help='0,1,2,3', required=True)
  parser.add_argument('--ngdType', type=str, help='max,sum.average', required=True)
  
  data_args = parser.parse_args()

  features = data_args.features
  
  ngdType = data_args.ngdType
  args.batch_size = 64
  print '---------------------------'
  print 'args.batch_size:',args.batch_size
  modelNEL = ctxSum(args,features)  #build named entity linking models
  train_batch_size,train_input, train_out, train_ent_mention_index, train_ent_mention_link_feature, \
  train_ent_mention_tag, train_ent_relcoherent,train_ent_linking_type,\
  train_ent_linking_candprob,train_ent_surfacewordv_feature = getData("kbp_training")
  print 'train_input shape:',np.shape(train_input)
  train_example = np.shape(train_input)[0]
  
  train_ent_relcoherent_ngd = train_ent_relcoherent[0]
  train_ent_relcoherent_fb =  train_ent_relcoherent[1]
  
  train_ent_relcoherent_fb_i_list=[]
  for i in range(len(train_batch_size)):
    train_ent_relcoherent_fb_i = np.zeros((train_ent_relcoherent_fb[i][2]))
    #print 'train_ent_relcoherent_fb_i:',np.shape(train_ent_relcoherent_fb_i)
    #there exist something wrong!
    for j in range(len(train_ent_relcoherent_fb[i][0])):
      items = train_ent_relcoherent_fb[i][0][j]
      train_ent_relcoherent_fb_i[items[0],items[1]]= train_ent_relcoherent_fb[i][1][j]
    train_ent_relcoherent_fb_i_list.append(train_ent_relcoherent_fb_i)

  val_batch_size,val_input, val_out, val_ent_mention_index, val_ent_mention_link_feature, \
  val_ent_mention_tag, val_ent_relcoherent,val_ent_linking_type,\
  val_ent_linking_candprob,val_ent_surfacewordv_feature =  getData("kbp_evaluation")
  print 'val_input shape:',np.shape(val_input) 
  val_ent_relcoherent_ngd = val_ent_relcoherent[0]
  val_ent_relcoherent_fb =  val_ent_relcoherent[1]
  val_example = np.shape(val_input)[0]
  
  val_ent_relcoherent_fb_i_list = []
  for i_val in range(len(val_batch_size)):
    val_ent_relcoherent_fb_i = np.zeros((val_ent_relcoherent_fb[i_val][2]))
          
    for j_val in range(len(val_ent_relcoherent_fb[i_val][0])):
      items = val_ent_relcoherent_fb[i_val][0][j_val]
      val_ent_relcoherent_fb_i[items[0],items[1]]= val_ent_relcoherent_fb[i_val][1][j_val]
      
    val_ent_relcoherent_fb_i_list.append(val_ent_relcoherent_fb_i)
  
  
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  
  with tf.Session(config=config) as sess:
    optimizer = tf.train.AdamOptimizer(0.01)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.005
    loss_linking = modelNEL.linking_loss  + lossL2
    print 'tvars_linking:',tvars
    grads,_ = tf.clip_by_global_norm(tf.gradients(loss_linking,tvars),10)
    train_op_linking = optimizer.apply_gradients(zip(grads,tvars))
    sess.run(tf.global_variables_initializer())
    
    if modelNEL.load(sess,args.restore,"KBP_"+features+"_"+ngdType):
      print "[*] ctxSum is loaded..."
    else:
      print "[*] There is no checkpoint for ctxSum"
    max_accracy_test=0
    #_,lstm_output_val = nerInstance.getEntityRecognition(val_input,val_out,'val')
    for e in range(200):
      print 'Epoch: %d------------' %(e)
      id_epoch=-1
      ptr = 0
      #we need to feed one document per epoch
      for i in range(len(train_batch_size)):
        train_ent_relcoherent_fb_i = train_ent_relcoherent_fb_i_list[i]
        if i==0:
          ptr =0
        else:
          ptr += train_batch_size[i-1]
        id_epoch += 1
        
        
        if id_epoch%10==0:
          accuracy_test_list=[];ent_mention_linking_tag_lists=[]; pred_lists = []
          for i_val in range(len(val_batch_size)):
            val_ent_relcoherent_fb_i = val_ent_relcoherent_fb_i_list[i_val]
            if i_val==0:
              ptr_val =0
            else:
              ptr_val += val_batch_size[i_val-1]
            
            #val_entMentIndex,val_entMent_length = genEntMentMask(args,np.shape(val_input)[0],val_ent_mention_index)
            val_entMentIndex,val_entMent_length = genEntMentMask(args,np.shape(val_input[ptr_val:min(val_example,ptr_val+val_batch_size[i_val])])[0],val_ent_mention_index[ptr_val:min(val_example,ptr_val+val_batch_size[i_val])]) 
            
            ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
             ent_surfacewordv_feature= getLinkingFeature(args,val_ent_mention_index,val_ent_mention_tag,
                                                  val_ent_mention_link_feature,val_ent_linking_type,
                                                  val_ent_linking_candprob,val_ent_surfacewordv_feature,val_batch_size[i_val],ptr_val,flag='kbp_val')
            loss2,lossl2,accuracy,pred,w1,w2,w3,w4,w5 = sess.run([loss_linking,lossL2,modelNEL.accuracy,modelNEL.prediction,modelNEL.w1,modelNEL.w2,modelNEL.w3,modelNEL.w4,modelNEL.w5],
                                     {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelNEL.candidate_ent_coherent_feature_ngd:val_ent_relcoherent_fb_i,
                                  modelNEL.candidate_ent_coherent_feature_fb:val_ent_relcoherent_fb_i,
                                  modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelNEL.input_data:val_input[ptr_val:min(val_example,ptr_val+val_batch_size[i_val])],
                                  modelNEL.entMentIndex:val_entMentIndex,
                                  #modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature, 
                                  modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                  modelNEL.keep_prob:1
                                     })
            if ptr_val==0:
              ent_mention_linking_tag_lists=ent_mention_linking_tag_list;
              pred_lists = pred
            else:
              ent_mention_linking_tag_lists = np.concatenate((ent_mention_linking_tag_lists,ent_mention_linking_tag_list))
              pred_lists = np.concatenate((pred_lists,pred))
            accuracy_test_list.append(accuracy)
          accuracy = np.average(accuracy_test_list)
          if accuracy > max_accracy_test:
            max_accracy_test = accuracy
            cPickle.dump(pred_lists,open('data/kbp/2014/evaluation/features/'+'50'+'/entityLinkingResult.p'+features+"_"+ngdType,'wb'))
            f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_lists,1),np.argmax(pred_lists,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_lists,1),np.argmax(pred_lists,1),average='macro')
            print '---------------'
            print 'test loss:',loss2,lossl2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
            print '---------------'
   
        train_entMentIndex,train_entMent_length = genEntMentMask(args,np.shape(train_input[ptr:min(train_example,ptr+train_batch_size[i])])[0],train_ent_mention_index[ptr:min(train_example,ptr+train_batch_size[i])]) 
        
        
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
        ent_surfacewordv_feature= getLinkingFeature(args,train_ent_mention_index,train_ent_mention_tag,
                                                train_ent_mention_link_feature,train_ent_linking_type,
                                                train_ent_linking_candprob,train_ent_surfacewordv_feature,train_batch_size[i],ptr,flag='kbp_train')
        if len(ent_mention_linking_tag_list)==0:
          continue
        _,loss2,accuracy,pred = sess.run([train_op_linking,loss_linking,modelNEL.accuracy,modelNEL.prediction],
                               {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                modelNEL.candidate_ent_coherent_feature_ngd:train_ent_relcoherent_fb_i,
                                modelNEL.candidate_ent_coherent_feature_fb:train_ent_relcoherent_fb_i,
                                modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                modelNEL.input_data:train_input[ptr:min(train_example,ptr+train_batch_size[i])],
                                modelNEL.entMentIndex:train_entMentIndex,
                                #modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature, 
                                modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                modelNEL.keep_prob:0.5
                               })
        
        f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
        print 'id_epoch:',id_epoch,'train loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
if __name__=="__main__":
  tf.app.run()

