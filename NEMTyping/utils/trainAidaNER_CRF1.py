# -*- coding: utf-8 -*-
'''
@time: 2016/12/20
@editor: wujs
@function: we add entity linking module
first version: we sum description sentence for candidates entities!
@revise: 2017/1/4
'''
import sys
import gensim
sys.path.append('utils')
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
import numpy as np
import tensorflow as tf
from model import seqLSTM_CRF
from utils import nerInputUtils as inputUtils
from embedding import WordVec,MyCorpus,RandomVec,get_input_conll2003,get_input_conll2003_test,get_input_figer_chunk_test_ner,get_input_figer,get_input_figer_chunk_train_ner
from evals import f1_chunk
import pprint
import time
import cPickle
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",50,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","conll2003_OntoNotes","dataset name")
flags.DEFINE_integer("sentence_length",250,"max sentence length")
flags.DEFINE_integer("class_size",3,"number of classes")
flags.DEFINE_integer("rnn_size",256,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",310,"hidden dimension of rnn")
flags.DEFINE_integer("model_dim",300,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","300","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",1,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.005,"apply dropout during training")
args = flags.FLAGS

def getCRFRet(tf_unary_scores,tf_transition_params,y,sequence_lengths):
  correct_labels = 0
  total_labels = 0
  predict = []
  for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,sequence_lengths):
    # Remove padding from the scores and tag sequence.
    tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
    y_ = y_[:sequence_length_]

    # Compute the highest scoring sequence.
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
        tf_unary_scores_, tf_transition_params)
    predict.append(viterbi_sequence)
    #print viterbi_sequence
    # Evaluate word-level accuracy.
    correct_labels += np.sum(np.equal(viterbi_sequence, y_))
    total_labels += sequence_length_
  accuracy = 100.0 * correct_labels / float(total_labels)
  
  return np.array(predict),accuracy



def main(_):
  pp.pprint(flags.FLAGS.__flags)

  '''
  @function: load the train and test datasets
  @entlinking context: 'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag
  '''
  model = seqLSTM_CRF(args)
 
  start_time = time.time()

  print 'start to load word2vec models!'
  word2vecModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time
  
#  WebQuestionUtils_train = inputUtils(args.rawword_dim,'data/WebQuestion/nerFeatures/',"train")
#  WebQuestion_input_train = np.asarray(WebQuestionUtils_train.emb);
#  WebQuestion_out_train =  np.argmax(np.asarray(WebQuestionUtils_train.tag),2)
#  WebQuestion_num_example_train = np.shape(WebQuestion_input_train)[0]
  
  print 'start to load data!'
  stime = time.time()
  testa_input,testa_out= get_input_conll2003_test(word2vecModel, args.model_dim, open('data/conll2003/testa.out'),
            "","",sentence_length=args.sentence_length)
  testa_num_example = np.shape(testa_input)[0]
  testa_out = np.argmax(testa_out,2)
  print 'laod testa time:',time.time()-stime
  
  stime = time.time()
  testb_input,testb_out= get_input_conll2003_test(word2vecModel, args.model_dim, open('data/conll2003/testb.out'),
            "","",sentence_length=args.sentence_length)
  testb_num_example = np.shape(testb_input)[0]
  testb_out = np.argmax(testb_out,2)
  print 'load testb time:',time.time()-stime
  
  stime = time.time()                               
  ace_input,ace_out=get_input_figer_chunk_test_ner(word2vecModel,args.model_dim, 
                                   open('data/ace/features/ace_Data.txt'),
                                   cPickle.load(open('data/ace/features/'+'ace_entMents.p','rb')), 
                                   "", "", sentence_length=args.sentence_length)
  ace_out = np.argmax(ace_out,2)
  ace_num_example = np.shape(ace_input)[0]
  print 'load ace time:',time.time()-stime
                                    
  stime = time.time()
  OntoNotes_input,OntoNotes_out=get_input_figer_chunk_test_ner(word2vecModel,args.model_dim, 
                                   open('data/OntoNotes/features/OntoNotesData_test.txt'),
                                   cPickle.load(open('data/OntoNotes/features/'+'test_entMents.p','rb')), 
                                   "", "", sentence_length=args.sentence_length)
  
  OntoNotes_num_example = np.shape(OntoNotes_input)[0]
  OntoNotes_out = np.argmax(OntoNotes_out,2)
  print 'load OntoNotes time:',time.time()-stime
      
  stime = time.time()
  WebQuestion_input,WebQuestion_out=get_input_figer_chunk_test_ner(word2vecModel,args.model_dim, 
                                   open('data/WebQuestion/features/test_Data.txt'),
                                   cPickle.load(open('data/WebQuestion/features/'+'test_entMents.p','rb')), 
                                   "", "", sentence_length=args.sentence_length)
  
  WebQuestion_num_example = np.shape(WebQuestion_input)[0]
  WebQuestion_out = np.argmax(WebQuestion_out,2)
  print 'load WebQuestion time:',time.time()-stime
  
  
  stime = time.time()
  figer_input,figer_out = get_input_figer(word2vecModel, args.model_dim, 
                                          open('data/figer_test/figerData.txt'),
                                          "","",sentence_length=args.sentence_length)
  figer_num_example = np.shape(figer_input)[0]
  figer_out = np.argmax(figer_out,2)
  print 'load figer time:',time.time()-stime
#  trainUtils = inputUtils(args.rawword_dim,dir_path,"train")
#  train_input = np.asarray(trainUtils.emb);train_out = np.argmax(np.asarray(trainUtils.tag),2)
#  print np.shape(train_input),np.shape(train_out)
  
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=8,inter_op_parallelism_threads=8)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time

  optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  #optimizer = tf.train.AdamOptimizer(args.learning_rate)
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  print 'tvars:',tvars
  #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.02
  totalLoss = model.loss #+ lossL2     

  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,"conll2003_OntoNotes"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for conll2003_OntoNotes"

  id_epoch = 0
  '''
  @train named entity recognition models
  '''
  maximum=0
  lossValidation = 1000000.0
  increase_times = 0
  minLoss = 10000
  for e in range(args.epoch):
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    for train_input,train_out in get_input_figer_chunk_train_ner(word2vecModel,args.model_dim,args.batch_size,
                                   open('data/OntoNotes/features/OntoNotesData_train.txt'),
                                   cPickle.load(open('data/OntoNotes/features/'+'train_entMents.p','rb')), 
                                   "", "", sentence_length=args.sentence_length):
      loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                       {model.input_data:testa_input,
                        model.output_data:testa_out,
                        model.num_examples:testa_num_example,
                        model.keep_prob:1})
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testa_out,length)
      precision,recall,fscore = f1_chunk('CRF',args, pred, testa_out, length)
      
      
      if lossValidation <= 0.1 or tloss > lossValidation:
        increase_times += 1
     
      lossValidation = tloss  
      m = fscore
      if m > maximum and tloss< minLoss:  #还得加上一个条件，loss也必须相较上一次有降低才能ok啦！这样对其他数据集才比较公平！
        maximum = m
        minLoss = tloss
        if maximum > 0.90:
          model.save(sess,args.restore,"conll2003_OntoNotes") #optimize in the dev file!
      #if id_epoch%20 == 0:
        print 'increase_times:',increase_times
        print("testa: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f  NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
        loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                       {model.input_data:testb_input,
                        model.output_data:testb_out,
                        model.num_examples:testb_num_example,
                        model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testb_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, testb_out, length)
        params = {'length':length,'args:':args,'testb_out':testb_out}
        
        cPickle.dump(params,open('data/figer/nerFeatures/figer_testb_NERret.p1','wb'))
        print("testb: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f  NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
      
      
        loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                         {model.input_data:figer_input,
                          model.output_data:figer_out,
                          model.num_examples:figer_num_example,
                          model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,figer_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, figer_out, length)
        params = {'length':length,'pred':pred,'target':figer_out,'args':args}
        cPickle.dump(params,open('data/figer_test/nerFeatures/figer_NERret.p1','wb'))
        #if fscore> maximum_figer:
        #  maximum_figer = fscore
        print("figer: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
        
        loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                         {model.input_data:WebQuestion_input,
                          model.output_data:WebQuestion_out,
                          model.num_examples:WebQuestion_num_example,
                          model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,WebQuestion_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, WebQuestion_out, length)
        params = {'length':length,'pred':pred,'target':figer_out,'args':args}
        cPickle.dump(params,open('data/WebQuestion/nerFeatures/WebQuestion_NERret.p1','wb'))
        #if fscore> maximum_figer:
        #  maximum_figer = fscore
        print("WebQuestion: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
        print "--------------------------------------------"
        loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                         {model.input_data:OntoNotes_input,
                          model.output_data:OntoNotes_out,
                          model.num_examples:OntoNotes_num_example,
                          model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,OntoNotes_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, OntoNotes_out, length)
        params = {'length':length,'pred':pred,'target':figer_out,'args':args}
        cPickle.dump(params,open('data/OntoNotes/nerFeatures/OntoNotes_NERret.p1','wb'))
        #if fscore> maximum_figer:
        #  maximum_figer = fscore
        print("OntoNotes: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
        print "--------------------------------------------"
        
      train_out = np.asarray(train_out,dtype=np.float32)
      train_out = np.argmax(train_out,2)
      
      _,loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                        {model.input_data:train_input,
                         model.output_data:train_out,
                         model.num_examples: np.shape(train_out)[0],
                         model.keep_prob:0.5})
      id_epoch += 1
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,train_out,length)
      precision,recall,fscore = f1_chunk('CRF',args, pred,train_out,length)
      if id_epoch %10 == 0:
        print("train: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
        _,loss1,length,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,model.length,model.unary_scores,model.transition_params],
                     {model.input_data:ace_input,
                      model.output_data:ace_out,
                      model.num_examples:ace_num_example,
                      model.keep_prob:0.5})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,ace_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, ace_out, length)
        print("ace: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
        print("--------------------------------------------")
        
    for kepoch in range(2):    
      for train_input,train_out in get_input_conll2003(word2vecModel,args.batch_size,300, open('data/conll2003/train.out'), sentence_length=args.sentence_length):
        loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                         {model.input_data:testa_input,
                          model.output_data:testa_out,
                          model.num_examples:testa_num_example,
                          model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testa_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred, testa_out, length)
        
        
        if lossValidation <= 0.1 or tloss > lossValidation:
          increase_times += 1
       
        lossValidation = tloss  
        m = fscore
        if m > maximum and tloss< minLoss:  #还得加上一个条件，loss也必须相较上一次有降低才能ok啦！这样对其他数据集才比较公平！
          maximum = m
          minLoss = tloss
          if maximum > 0.90:
            model.save(sess,args.restore,"conll2003_OntoNotes") #optimize in the dev file!
        #if id_epoch%20 == 0:
          print 'increase_times:',increase_times
          print("testa: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f  NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
          loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                         {model.input_data:testb_input,
                          model.output_data:testb_out,
                          model.num_examples:testb_num_example,
                          model.keep_prob:1})
          pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testb_out,length)
          precision,recall,fscore = f1_chunk('CRF',args, pred, testb_out, length)
          params = {'length':length,'args:':args,'testb_out':testb_out}
          
          cPickle.dump(params,open('data/figer/nerFeatures/figer_testb_NERret.p1','wb'))
          print("testb: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f  NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
        
        
          loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                           {model.input_data:figer_input,
                            model.output_data:figer_out,
                            model.num_examples:figer_num_example,
                            model.keep_prob:1})
          pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,figer_out,length)
          precision,recall,fscore = f1_chunk('CRF',args, pred, figer_out, length)
          params = {'length':length,'pred':pred,'target':figer_out,'args':args}
          cPickle.dump(params,open('data/figer_test/nerFeatures/figer_NERret.p1','wb'))
          #if fscore> maximum_figer:
          #  maximum_figer = fscore
          print("figer: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
          
          loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                           {model.input_data:WebQuestion_input,
                            model.output_data:WebQuestion_out,
                            model.num_examples:WebQuestion_num_example,
                            model.keep_prob:1})
          pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,WebQuestion_out,length)
          precision,recall,fscore = f1_chunk('CRF',args, pred, WebQuestion_out, length)
          params = {'length':length,'pred':pred,'target':figer_out,'args':args}
          cPickle.dump(params,open('data/WebQuestion/nerFeatures/WebQuestion_NERret.p1','wb'))
          #if fscore> maximum_figer:
          #  maximum_figer = fscore
          print("WebQuestion: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
          print "--------------------------------------------"
          loss1,length,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.unary_scores,model.transition_params],
                           {model.input_data:OntoNotes_input,
                            model.output_data:OntoNotes_out,
                            model.num_examples:OntoNotes_num_example,
                            model.keep_prob:1})
          pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,OntoNotes_out,length)
          precision,recall,fscore = f1_chunk('CRF',args, pred, OntoNotes_out, length)
          params = {'length':length,'pred':pred,'target':figer_out,'args':args}
          cPickle.dump(params,open('data/OntoNotes/nerFeatures/OntoNotes_NERret.p1','wb'))
          #if fscore> maximum_figer:
          #  maximum_figer = fscore
          print("OntoNotes: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
          print "--------------------------------------------"
          
        train_out = np.asarray(train_out,dtype=np.float32)
        train_out = np.argmax(train_out,2)
        
        _,loss1,tloss,length,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,totalLoss,model.length,model.unary_scores,model.transition_params],
                          {model.input_data:train_input,
                           model.output_data:train_out,
                           model.num_examples: np.shape(train_out)[0],
                           model.keep_prob:0.5})
        id_epoch += 1
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,train_out,length)
        precision,recall,fscore = f1_chunk('CRF',args, pred,train_out,length)
        if id_epoch %10 == 0:
          print("train: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
          _,loss1,length,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,model.length,model.unary_scores,model.transition_params],
                       {model.input_data:ace_input,
                        model.output_data:ace_out,
                        model.num_examples:ace_num_example,
                        model.keep_prob:0.5})
          pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,ace_out,length)
          precision,recall,fscore = f1_chunk('CRF',args, pred, ace_out, length)
          print("ace: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
          print("--------------------------------------------")
    
    
#  except:
#    print 'finished train'
if __name__=='__main__':
  tf.app.run()
