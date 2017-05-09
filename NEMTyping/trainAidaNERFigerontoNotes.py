# -*- coding: utf-8 -*-
'''
@time: 2016/12/20
@editor: wujs
@function: we add entity linking module
first version: we sum description sentence for candidates entities!
@revise: 2017/1/4
'''
import sys
sys.path.append('utils')
import gensim
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
import random_vec
import numpy as np
import tensorflow as tf
from model import seqMLP,seqCtxLSTM,seqLSTM
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk,get_input_figer_chunk_train,get_input_figerTest_chunk
import cPickle
from utils import nerInputUtils as inputUtils
import pprint
import time
from scipy.sparse import coo_matrix
#dataset = "figer_MLP_test"
dataset = "OntoNotes"

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",1500,"batch size of training")
flags.DEFINE_string("datasets",dataset,"dataset name")
flags.DEFINE_integer("sentence_length",250,"max sentence length")
flags.DEFINE_integer("class_size",89,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",311,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.005,"apply dropout during training")
args = flags.FLAGS

def getAccuracy(predArray,targetArray):
  right = 0; alls = 0
  '''
  @revise the accuracy test method
  '''
  precision=[]
  recall=[]
  for i in range(len(predArray)):
    pred = predArray[i]
    target = targetArray[i]
    
    lents = len(np.nonzero(target)[0])*(-1)
    
    predRet = set(np.argsort(pred)[lents:])
    targetRet = set(np.argsort(target)[lents:])
    
    rightset = predRet & targetRet
    if len(rightset)==0:
      precision.append(0.0)
    else:
      precision.append(len(rightset)*1.0/len(predRet))
    
    recall.append(len(rightset) *1.0/len(targetRet))
  
    if predRet == targetRet:
      right += 1
    alls += 1
    
  return 1.0 * right/alls * 100,precision,recall
    
'''
@sentence_final: shape:(batch_size,sequence_length,dims)
'''
def padZeros(sentence_final,max_sentence_length=80,dims=111):
  for i in range(len(sentence_final)):
    offset = max_sentence_length-len(sentence_final[i])
    sentence_final[i] += [[0]*dims]*offset
    
  return np.asarray(sentence_final)

def genEntCtxMask(batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  entCtxLeft_masks=[]
  entCtxRight_masks=[]
  for i in range(entNums):
    items = entment_mask_final[i]
    ids = items[0];start=items[1];end=items[2]
    temp_entCtxLeft_mask=[];temp_entCtxRight_mask = []
    left = max(0,start-10); 
    right = min(args.sentence_length,end+10)
    for ient in range(left,start):
        temp_entCtxLeft_mask.append(ids*args.sentence_length+ient)
    for ient in range(end,right):
        temp_entCtxRight_mask.append(ids*args.sentence_length+ient)
        
    if start-left < 10:
      temp_entCtxLeft_mask+= [batch_size*args.sentence_length] * (10-(start-left))
    
    if right-end < 10:
      temp_entCtxRight_mask+= [batch_size*args.sentence_length] * (10-(right-end))
      
    entCtxLeft_masks.append(temp_entCtxLeft_mask)
    entCtxRight_masks.append(temp_entCtxRight_mask)
  return entCtxLeft_masks,entCtxRight_masks
    
def genEntMentMask(batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  entment_masks = []
  #need to limit the length of the entity mentions
  for i in range(entNums):
    items = entment_mask_final[i]
    ids = items[0];start=items[1];end=items[2]
    temp_entment_masks=[]
    for ient in range(start,end):
        temp_entment_masks.append(ids*args.sentence_length+ient)
    if end-start <5:
      temp_entment_masks+= [batch_size*args.sentence_length] * (5-(end-start))
    if end-start > 5:
      temp_entment_masks = temp_entment_masks[0:5]
    entment_masks.append(list(temp_entment_masks))
  return np.asarray(entment_masks)
    
def main(_):
  pp.pprint(flags.FLAGS.__flags)

  '''
  @function: load the train and test datasets
  @entlinking context: 'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag
  '''
  model = seqLSTM(args)
  print 'start to load data!'
  start_time = time.time()
  #word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
  #word2vecModel = None
  word2vecModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time
  
  test_entment_mask,test_sentence,test_tag = get_input_figerTest_chunk('LSTM',dataset,'data/'+dataset+'/',model=word2vecModel,word_dim=args.word_dim,sentence_length=250)
  test_size = len(test_sentence)
  print 'test_size',test_size
  test_sentence += [[[0]*args.word_dim]*args.sentence_length]*(args.batch_size-test_size)
  print 'test_sentence shape:',np.shape(test_sentence)
#  start_time = time.time()                                           
#  testa_entment_mask,testa_sentence_final,testa_tag_final = get_input_figer_chunk(args.batch_size,'data/figer/',"testa",model=word2vecModel,word_dim=100,sentence_length=80)
#  print 'load testa data cost time:',time.time()-start_time                                            
#  start_time = time.time()           
#  testb_entment_mask,testb_sentence_final,testb_tag_final = get_input_figer_chunk(args.batch_size,'data/figer/',"testb",model=word2vecModel,word_dim=100,sentence_length=80)
#  print 'load testb data cost time:',time.time()-start_time
                                            
                                              
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time

  #optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  optimizer = tf.train.AdamOptimizer(args.learning_rate)
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  print 'tvars:',tvars
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.001  #parameter has a very important effect on training!
  totalLoss = model.loss  + lossL2                
#  start_time = time.time()
#  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
#  train_op = optimizer.apply_gradients(zip(grads, tvars))
  train_op = optimizer.minimize(totalLoss)
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,dataset):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for figer_MLP_test"

  id_epoch = 0
  maximum=0
  '''
  @train named entity recognition models
  '''
  for epoch in range(10):
    print 'epoch:',epoch
    print '---------------------------------'
    for train_entment_mask,train_sentence_final,train_tag_final in get_input_figer_chunk_train('LSTM',dataset,args.batch_size,'data/'+dataset+'/',"train",model=word2vecModel,word_dim=args.word_dim,sentence_length=args.sentence_length):
      
#      if id_epoch % 50 ==0:
#        accuracy_list=[]
#        for i in range(len(testa_sentence_final)):
          #testa_input = padZeros(testa_sentence_final[i])
#          testa_input = np.asarray(testa_sentence_final[i],dtype=np.float32)
#          testa_out = testa_tag_final[i]
#          testa_entMentIndex = genEntMentMask(args.batch_size,testa_entment_mask[i])
#          num_examples = len(testa_entMentIndex)
#          type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
#          loss1,tloss,length,pred,target = sess.run([model.loss,totalLoss,model.length,model.prediction,model.dense_outputdata],
#                         {model.input_data:testa_input,
#                          model.output_data:tf.SparseTensorValue(testa_out[0],testa_out[1],type_shape),
#                          model.entMentIndex:testa_entMentIndex,
#                          model.keep_prob:1})
#          accuracy = getAccuracy(pred,target)
#          accuracy_list.append(accuracy)
        
#        if np.average(accuracy_list) > maximum:
#          maximum = np.average(accuracy_list)
#          if maximum > 63.0:
#            model.save(sess,args.restore,"figer") #optimize in the dev file!
#          print "------------------"
#          print("testa: loss:%.4f total loss:%.4f average accuracy:%.6f" %(loss1,tloss,maximum))
#          accuracy_list=[]
#          for i in range(len(testb_sentence_final)):
#            #testb_input = padZeros(testb_sentence_final[i])
#            testb_input = np.asarray(testb_sentence_final[i],dtype=np.float32)
#            testb_out = testb_tag_final[i]
#            testb_entMentIndex = genEntMentMask(args.batch_size,testb_entment_mask[i])
#            num_examples = len(testb_entMentIndex)
#            type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
#            loss1,tloss,length,pred,target = sess.run([model.loss,totalLoss,model.length,model.prediction,model.dense_outputdata],
#                       {model.input_data:testb_input,
#                        model.output_data:tf.SparseTensorValue(testb_out[0],testb_out[1],type_shape),
#                        model.entMentIndex:testb_entMentIndex,
#                        model.keep_prob:1})
#            accuracy = getAccuracy(pred,target)
#            accuracy_list.append(accuracy)
#          print("testb: loss:%.4f total loss:%.4f average accuracy:%.6f" %(loss1,tloss,np.average(accuracy_list)))
      if id_epoch % 10==0:
        test_input = np.asarray(test_sentence,dtype=np.float32)
        print np.shape(test_input)
        test_out = test_tag
        test_entMentIndex = genEntMentMask(np.shape(test_input)[0],test_entment_mask)
        test_entCtxLeft_Index,test_entCtxRight_Index = genEntCtxMask(np.shape(test_input)[0],test_entment_mask)
        num_examples = len(test_entMentIndex)
        type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
        loss1,tloss,pred,target = sess.run([model.loss,totalLoss,model.prediction,model.dense_outputdata],
                       {model.input_data:test_input,
                        model.output_data:tf.SparseTensorValue(test_out[0],test_out[1],type_shape),
                        model.entMentIndex:test_entMentIndex,
                        model.entCtxLeftIndex:test_entCtxLeft_Index,
                        model.entCtxRightIndex:test_entCtxRight_Index,
                        model.keep_prob:1})
        accuracy,precision,recall = getAccuracy(pred,target)
        if (np.average(precision)+np.average(recall))==0:
          f1 = 0
        else:
          f1 = np.average(precision)*np.average(recall)*2/(np.average(precision)+np.average(recall))*100
        if f1 > maximum:
          maximum = f1
          if maximum > 63.0:
            model.save(sess,args.restore,dataset) #optimize in the dev file!
          cPickle.dump(pred,open('data/'+dataset+'/fulltypeFeatures/'+'testType.p','wb'))
          print("test: loss:%.4f total loss:%.4f average accuracy:%.4f F1:%.4f" %(loss1,tloss,accuracy,f1))
          print "------------------"
      
      #train_input = padZeros(train_sentence_final)
      train_input = np.asarray(train_sentence_final,dtype=np.float32)
      train_entMentIndex = genEntMentMask(args.batch_size,train_entment_mask)
      train_entCtxLeft_Index,train_entCtxRight_Index = genEntCtxMask(args.batch_size,train_entment_mask)
      train_out = train_tag_final #we need to generate entity mention masks!
      num_examples = len(train_entMentIndex)
      type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
      _,loss1,tloss,pred,target = sess.run([train_op,model.loss,totalLoss,model.prediction,model.dense_outputdata],
                        {model.input_data:train_input,
                         model.output_data:tf.SparseTensorValue(train_out[0],train_out[1],type_shape),
                         model.entMentIndex:train_entMentIndex,
                         model.entCtxLeftIndex:train_entCtxLeft_Index,
                         model.entCtxRightIndex:train_entCtxRight_Index,
                         model.keep_prob:0.5})
      id_epoch += 1
      
      if id_epoch % 100 == 0:
        accuracy,precision,recall = getAccuracy(pred,target)
        if (np.average(precision)+np.average(recall))==0:
          f1 = 0
        else:
          f1 = np.average(precision)*np.average(recall)*2/(np.average(precision)+np.average(recall))*100
        print("ids: %d,train: loss:%.4f total loss:%.4f accuracy:%.4f  F1:%.4f" %(id_epoch,loss1,tloss,accuracy,f1))
if __name__=='__main__':
  tf.app.run()
