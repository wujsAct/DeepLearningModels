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
from model import seqMLP,seqCtxLSTM,seqLSTM,seqCNN
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk_train,get_input_figerTest_chunk
import cPickle
from evals import getTypeEval,getRelTypeEval
from utils import genEntCtxMask,genEntMentMask
import pprint
import time
import argparse

'''
只能把argument放到这种位置了！
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='[figer, OntoNotes]', required=True)
parser.add_argument('--wordEmbed', type=str, help='[LSTM,MLP]', required=True)
parser.add_argument('--batch_size', type=int, help='[1000 for figer; 256 for OntoNotes]', required=True)
parser.add_argument('--class_size', type=int, help='[1000 for figer; 1500 for OntoNotes]', required=True)
parser.add_argument('--model', type=str, help='[0:seqCNN,1:seqMLP,2:seqCtxLSTM,3:seqLSTM]', required=True)
parser.add_argument('--sentence_length', type=int, help='[250 for OntoNotes; 62 for figer]', required=True)
parser.add_argument('--iterateEpoch', type=int, help='[10 for figer, 1 for OntoNotes]', required=True)
parser.add_argument('--learning_rate', type=float, help='[0.005 for figer, 0.001 for OntoNotes]', required=True)
parser.add_argument('--threshold', type=float, help='[0.6 for pcnn, 0.5 for OntoNotes]', required=True)
'''
@try to utilize the largest and small learning rates to figure out the property 
'''
args = parser.parse_args()

dataset = args.dataset
wordEmbed = args.wordEmbed
batch_size = args.batch_size
class_size = args.class_size
type_model =  args.model
sentence_length = args.sentence_length
iterateEpoch = args.iterateEpoch
learning_rate = args.learning_rate
threshold = args.threshold
print dataset
print wordEmbed
print sentence_length

word_dims = 300
hidden_size = 256

if dataset == 'OntoNotes':
  hidden_size = 128

valtest = 'data/'+dataset+'/'
#else:
#  valtest = 'corpus/Wiki/'
  
if wordEmbed =='LSTM':
  word_dims = 310
  

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",10,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",batch_size,"batch size of training")
flags.DEFINE_string("datasets",dataset,"dataset name")
flags.DEFINE_integer("sentence_length",sentence_length,"max sentence length")
flags.DEFINE_integer("class_size",class_size,"number of classes")
flags.DEFINE_integer("rnn_size",hidden_size,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",word_dims,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",learning_rate,"learning rates")
args = flags.FLAGS
    
def main(_):
  pp.pprint(flags.FLAGS.__flags)

  '''
  @function: load the train and test datasets
  @entlinking context: 'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag
  '''
  #model = seqCtxLSTM(args)
  if type_model=='0':
    model = seqCNN(args)
  elif type_model=='1':
    model = seqMLP(args)
  elif type_model=='2':
    model = seqCtxLSTM(args)
  else:
    model = seqLSTM(args)
    
                                              
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=8,inter_op_parallelism_threads=8)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time
  if type_model == '0' or type_model =='1':
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
  else:
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  print 'tvars:',tvars
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'biase' not in v.name])*0.02 #parameter has a very important effect on training!
  totalLoss = model.loss  + lossL2
  print totalLoss          
      
  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  #train_op = optimizer.minimize(totalLoss)
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,dataset):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for "+dataset
           
           
  print 'start to load data!'
  
  start_time = time.time()
  #word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
  #word2vecModel = None
  word2vecModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time

  '''
  #we need to consider different batch_size datasets
  '''
  
  testParams = get_input_figerTest_chunk(wordEmbed,'testb',valtest,args.batch_size,model=word2vecModel,word_dim=args.word_dim,sentence_length=args.sentence_length)
  
  start_time = time.time()                                           
  ValidtestParams = get_input_figerTest_chunk(wordEmbed,'testa',valtest,args.batch_size,model=word2vecModel,word_dim=args.word_dim,sentence_length=args.sentence_length)
  print 'load testa data cost time:',time.time()-start_time
                                              
  
  id_epoch = 0
  maximum=0;
  maximum_miF1=0;
  maximum_maF1=0;
  minLoss = sys.maxint
  '''
  @train named entity recognition models
  '''
  for epoch in range(200):
    print 'epoch:',epoch
    print '---------------------------------'
    
    for train_entment_mask,train_sentence_final,train_tag_final in get_input_figer_chunk_train(wordEmbed,dataset,args.batch_size,'data/'+dataset+'/',"train",model=word2vecModel,word_dim=args.word_dim,sentence_length=args.sentence_length):
      if id_epoch % iterateEpoch==0:
        '''
        @we also need a validation test to help to early stop the training process!
        '''
        test_right=0;test_alls=0;test_precision=[];test_recall=[];
        test_right_cross=[];test_predRet_list=[];test_targetRet_list=[]
        tloss_List = []
        loss_List = []
        for i in range(len(ValidtestParams)):
          test_entment_mask,test_sentence,test_tag = ValidtestParams[i]
          test_size = len(test_sentence)
          #print i,':validate_size',test_size
          #print np.shape(test_sentence)
          #print 'test_size',test_size
          if test_size < args.batch_size:
            test_sentence = np.concatenate((test_sentence,np.asarray([[[0]*args.word_dim]*args.sentence_length]*(args.batch_size-test_size))),0)
          #print 'test_sentence shape:',np.shape(test_sentence)
          test_input = np.asarray(test_sentence,dtype=np.float32)
          test_out = test_tag
          test_entMentIndex = genEntMentMask(args,np.shape(test_input)[0],test_entment_mask)
          test_entCtxLeft_Index,test_entCtxRight_Index = genEntCtxMask(args,np.shape(test_input)[0],test_entment_mask)
          num_examples = len(test_entMentIndex)
          type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
        
          pos1 = np.expand_dims(np.array(num_examples*[[0]*5],np.float32),-1)
          pos2 = np.expand_dims(np.array(num_examples*[range(-10,0,1)],np.float32),-1)
          pos3 = np.expand_dims(np.array(num_examples*[range(1,11)],np.float32),-1)
          
          loss1,tloss,pred,target = sess.run([model.loss,totalLoss,model.prediction,model.dense_outputdata],
                         {model.input_data:test_input,
                          model.output_data:tf.SparseTensorValue(test_out[0],test_out[1],type_shape),
                          model.entMentIndex:test_entMentIndex,
                          model.entCtxLeftIndex:test_entCtxLeft_Index,
                          model.entCtxRightIndex:test_entCtxRight_Index,
                          model.pos_f1:pos1,
                          model.pos_f2:pos2,
                          model.pos_f3:pos3,
                          model.keep_prob:1})
          right,alls,precision,recall,right_cross,predRet_list,targetRet_list = getTypeEval(threshold,pred,target)
          test_right += right; test_alls += alls
          test_precision+=precision;test_recall += precision
          test_right_cross += right_cross; test_targetRet_list+= targetRet_list; test_predRet_list += predRet_list
          tloss_List.append(tloss)
          loss_List.append(loss1)
          
        
        f1_strict,f1_macro,f1_micro = getRelTypeEval(test_right,test_alls,test_precision,test_recall,test_right_cross,test_predRet_list,test_targetRet_list)
        
        if np.average(tloss_List) < minLoss and (f1_strict > maximum): #or f1_macro > maximum_maF1 or f1_micro > maximum_miF1):
          maximum = max(f1_strict,maximum)
          #maximum_maF1 = max(f1_macro,maximum_maF1)
          #maximum_miF1 = max(f1_micro,maximum_miF1)
          minLoss = np.average(tloss_List)
          #if maximum > 53.0:
          #  model.save(sess,args.restore,dataset) #optimize in the dev file!
          print '-------------------------------------'
          print("test: loss:%.4f total loss:%.4f F1_strict:%.2f f1_macro:%.2f f1_micro:%.2f" %(np.average(loss_List),minLoss,f1_strict,f1_macro,f1_micro))
           
          
          test_right=0;test_alls=0;test_precision=[];test_recall=[];
          test_right_cross=[];test_predRet_list=[];test_targetRet_list=[]
          tloss_List = []
          loss_List = []
          for i in range(len(testParams)):
            test_entment_mask,test_sentence,test_tag = testParams[i]
            test_size = len(test_sentence)
            #print i,':test_size',test_size
            #print np.shape(test_sentence)
            #print 'test_size',test_size
            if test_size < args.batch_size:
              test_sentence = np.concatenate((test_sentence,np.asarray([[[0]*args.word_dim]*args.sentence_length]*(args.batch_size-test_size))),0)
            #print 'test_sentence shape:',np.shape(test_sentence)
            test_input = np.asarray(test_sentence,dtype=np.float32)
            test_out = test_tag
            test_entMentIndex = genEntMentMask(args,np.shape(test_input)[0],test_entment_mask)
            test_entCtxLeft_Index,test_entCtxRight_Index = genEntCtxMask(args,np.shape(test_input)[0],test_entment_mask)
            num_examples = len(test_entMentIndex)
            type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
          
            pos1 = np.expand_dims(np.array(num_examples*[[0]*5],np.float32),-1)
            pos2 = np.expand_dims(np.array(num_examples*[range(-10,0,1)],np.float32),-1)
            pos3 = np.expand_dims(np.array(num_examples*[range(1,11)],np.float32),-1)
            
            loss1,tloss,pred,target = sess.run([model.loss,totalLoss,model.prediction,model.dense_outputdata],
                           {model.input_data:test_input,
                            model.output_data:tf.SparseTensorValue(test_out[0],test_out[1],type_shape),
                            model.entMentIndex:test_entMentIndex,
                            model.entCtxLeftIndex:test_entCtxLeft_Index,
                            model.entCtxRightIndex:test_entCtxRight_Index,
                            model.pos_f1:pos1,
                            model.pos_f2:pos2,
                            model.pos_f3:pos3,
                            model.keep_prob:1})
            right,alls,precision,recall,right_cross,predRet_list,targetRet_list = getTypeEval(threshold,pred,target)
            test_right += right; test_alls += alls
            test_precision+=precision;test_recall += precision
            test_right_cross += right_cross; test_targetRet_list+= targetRet_list; test_predRet_list += predRet_list
            tloss_List.append(tloss)
            loss_List.append(loss1)
            
          
          f1_strict,f1_macro,f1_micro = getRelTypeEval(test_right,test_alls,test_precision,test_recall,test_right_cross,test_predRet_list,test_targetRet_list)
          if f1_strict >= 50.0:
            model.save(sess,args.restore,dataset) #optimize in the dev file!
          cPickle.dump(pred,open('data/'+dataset+'/fulltypeFeatures/'+type_model+'_test_type.p','wb'))
          print("testa: loss:%.4f total loss:%.4f F1_strict:%.2f f1_macro:%.2f f1_micro:%.2f" %(np.average(loss_List),np.average(tloss_List),f1_strict,f1_macro,f1_micro))
          print '----------------------------'  
          
      #exit(0)
  #train_input = padZeros(train_sentence_final)
      
      
      train_size = len(train_sentence_final)
      if train_size < args.batch_size:
        train_sentence_final = np.concatenate( (train_sentence_final,[[[0]*args.word_dim]*args.sentence_length]*(args.batch_size-train_size)),0)
      train_input = np.asarray(train_sentence_final,dtype=np.float32)
      train_entMentIndex = genEntMentMask(args,args.batch_size,train_entment_mask)
      train_entCtxLeft_Index,train_entCtxRight_Index = genEntCtxMask(args,args.batch_size,train_entment_mask)
      train_out = train_tag_final #we need to generate entity mention masks!
      num_examples = len(train_entMentIndex)
      pos1 = np.expand_dims(np.array(num_examples*[[0]*5],np.float32),-1)
      pos2 = np.expand_dims(np.array(num_examples*[range(-10,0,1)],np.float32),-1)
      pos3 = np.expand_dims(np.array(num_examples*[range(1,11)],np.float32),-1)
        
      type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
      _,loss1,tloss,pred,target = sess.run([train_op,model.loss,totalLoss,model.prediction,model.dense_outputdata],
                        {model.input_data:train_input,
                         model.output_data:tf.SparseTensorValue(train_out[0],train_out[1],type_shape),
                         model.entMentIndex:train_entMentIndex,
                         model.entCtxLeftIndex:train_entCtxLeft_Index,
                         model.entCtxRightIndex:train_entCtxRight_Index,
                         model.pos_f1:pos1,
                         model.pos_f2:pos2,
                         model.pos_f3:pos3,
                         model.keep_prob:0.5})
      id_epoch += 1
      
      if dataset=='figer':
        if id_epoch % 50 == 0:
          right,alls,precision,recall,right_cross,predRet_list,targetRet_list = getTypeEval(threshold,pred,target)
          f1_strict,f1_macro,f1_micro = getRelTypeEval(right,alls,precision,recall,right_cross,predRet_list,targetRet_list)
          print("ids: %d,train: loss:%.4f total loss:%.4f F1_strict:%.2f f1_macro:%.2f f1_micro:%.2f" %(id_epoch,loss1,tloss,f1_strict,f1_macro,f1_micro))
      else:
        if id_epoch % 20 == 0:
          right,alls,precision,recall,right_cross,predRet_list,targetRet_list = getTypeEval(threshold,pred,target)
          f1_strict,f1_macro,f1_micro = getRelTypeEval(right,alls,precision,recall,right_cross,predRet_list,targetRet_list)
          print("ids: %d,train: loss:%.4f total loss:%.4f F1_strict:%.2f f1_macro:%.2f f1_micro:%.2f" %(id_epoch,loss1,tloss,f1_strict,f1_macro,f1_micro))
          
if __name__=='__main__':
  tf.app.run()
