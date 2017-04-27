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
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
import numpy as np
import tensorflow as tf
from model import seqLSTM_CRF
from utils import nerInputUtils as inputUtils
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk_train_ner
from evals import f1_chunk
import pprint
import time
import cPickle
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","conll2003","dataset name")
flags.DEFINE_integer("sentence_length",124,"max sentence length")
flags.DEFINE_integer("class_size",3,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",111,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.01,"apply dropout during training")
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
  print 'start to load data!'
  start_time = time.time()
#  trainUtils = inputUtils(args.rawword_dim,"train")
#  train_TFfileName = trainUtils.TFfileName; train_nerShapeFile = trainUtils.nerShapeFile;
#  train_batch_size = args.batch_size;
  
  #word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
  
  
  dir_path ='data/conll2003/features/'
  testaUtils = inputUtils(args.rawword_dim,dir_path,"testa")
  testa_input = np.asarray(testaUtils.emb);testa_out =  np.argmax(np.asarray(testaUtils.tag),2)
  testa_num_example = np.shape(testa_input)[0]
  
#  print np.shape(testa_input),np.shape(testa_out)
#  testb_input = testa_input;testb_out=testa_out;testb_num_example=testa_num_example
#  
#  train_input = testa_input;train_out=testa_out;
  
  testbUtils = inputUtils(args.rawword_dim,dir_path,"testb")
  testb_input = np.asarray(testbUtils.emb);testb_out =  np.argmax(np.asarray(testbUtils.tag),2)
  testb_num_example = np.shape(testb_input)[0]
  print np.shape(testb_input),np.shape(testb_out)
  
  figerUtils = inputUtils(args.rawword_dim,'data/figer_test/features/',"figer")
  figer_input = np.asarray(figerUtils.emb);figer_out =  np.argmax(np.asarray(figerUtils.tag),2)
  figer_num_example = np.shape(figer_input)[0]
  print np.shape(figer_input),np.shape(figer_out)
  
  trainUtils = inputUtils(args.rawword_dim,dir_path,"train")
  train_input = np.asarray(trainUtils.emb);train_out = np.argmax(np.asarray(trainUtils.tag),2)
  print np.shape(train_input),np.shape(train_out)
  
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=8,inter_op_parallelism_threads=8)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time

  #optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  optimizer = tf.train.AdamOptimizer(args.learning_rate)
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='seqLSTM_variables')
  print 'tvars:',tvars
  
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.01
  totalLoss = model.loss + lossL2     

  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,"conll2003"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for conll2003"

  id_epoch = 0
  '''
  @train named entity recognition models
  '''
  maximum=0
  maximum_figer=0

#  for train_input,train_out in ner_read_TFRecord(sess,train_TFfileName,
#                                                 train_nerShapeFile,train_batch_size,args.epoch):
  for e in range(args.epoch):
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    for ptr in xrange(0,len(train_input),args.batch_size):
    #for train_input,train_out in get_input_figer_chunk_train_ner(args.batch_size,'data/figer/',"train",model=word2vecModel,word_dim=100,sentence_length=124):
      
      loss1,tloss,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.output,model.unary_scores,model.transition_params],
                       {model.input_data:testa_input,
                        model.output_data:testa_out,
                        model.num_examples:testa_num_example,
                        model.keep_prob:1})
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testa_out,length)
        
      fscore = f1_chunk('CRF',args, pred, testa_out, length)

      m = fscore
      if m > maximum:
        maximum = m
        if maximum > 0.80:
          model.save(sess,args.restore,"conll2003") #optimize in the dev file!
        print "----------------------------------"
        print("testa: loss:%.4f total loss:%.4f accuracy:%.4f NER:%.2f" %(loss1,tloss,accuracy,100*m))
        
        loss1,tloss,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,totalLoss,model.length,model.output,model.unary_scores,model.transition_params],
                       {model.input_data:testb_input,
                        model.output_data:testb_out,
                        model.num_examples:testb_num_example,
                        model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testb_out,length)
        fscore = f1_chunk('CRF',args, pred, testb_out, length)
        print("testb: loss:%.4f total loss:%.4f accuracy:%f NER:%.2f" %(loss1,tloss,accuracy,100*fscore))
      
      
        loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                         {model.input_data:figer_input,
                          model.output_data:figer_out,
                          model.num_examples:figer_num_example,
                          model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,figer_out,length)
        fscore = f1_chunk('CRF',args, pred, figer_out, length)
        if fscore> maximum_figer:
          maximum_figer = fscore
          print("figer: loss:%.4f accuracy:%f NER:%.2f" %(loss1,accuracy,100*fscore))
          print "--------------------------------------------"
      num_example = min(ptr+args.batch_size,len(train_input)) - ptr  
      batch_train_input = train_input[ptr:min(ptr+args.batch_size,len(train_input))]
      batch_train_out = train_out[ptr:min(ptr+args.batch_size,len(train_input))]
      _,loss1,tloss,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,totalLoss,model.length,model.output,model.unary_scores,model.transition_params],
                        {model.input_data:batch_train_input,
                         model.output_data:batch_train_out,
                         model.num_examples: num_example,
                         model.keep_prob:0.5})
#      train_out = np.argmax(train_out,2)
#      _,loss1,tloss,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,totalLoss,model.length,model.output,model.unary_scores,model.transition_params],
#                        {model.input_data:train_input,
#                         model.output_data:train_out,
#                         model.num_examples: args.batch_size,
#                         model.keep_prob:0.5})
      id_epoch += 1
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,batch_train_out,length)
      fscore = f1_chunk('CRF',args, pred,batch_train_out,length)
      if id_epoch %10==0:
        print("train: loss:%.4f total loss:%.4f accuracy:%f NER:%.2f" %(loss1,tloss,accuracy,100*fscore))
#  except:
#    print 'finished train'
if __name__=='__main__':
  tf.app.run()
