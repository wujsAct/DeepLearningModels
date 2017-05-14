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
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk_test_ner_train
from evals import f1_chunk
import pprint
import time
import cPickle
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",32,"batch size of training")
flags.DEFINE_string("datasets","WebQuestion","dataset name")
flags.DEFINE_integer("sentence_length",124,"max sentence length")
flags.DEFINE_integer("class_size",3,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",310,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","300","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
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

  #word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
  print 'start to load word2vec models!'
  #trained_model = cPickle.load(open(args.use_model, 'rb'))
  word2vecModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time
  
  print 'start to load data!'
  WebQuestionUtils = inputUtils(args.rawword_dim,'data/WebQuestion/nerFeatures/',"test")
  WebQuestion_input = np.asarray(WebQuestionUtils.emb);WebQuestion_out =  np.argmax(np.asarray(WebQuestionUtils.tag),2)
  WebQuestion_num_example = np.shape(WebQuestion_input)[0]
  print np.shape(WebQuestion_input),np.shape(WebQuestion_out)
  
  
  
  
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
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.01
  totalLoss = model.loss + lossL2     

  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,"WebQuestion"):
    print "[*] WebQuestion is loaded..."
  else:
    print "[*] There is no checkpoint for WebQuestion"

  id_epoch = 0
  '''
  @train named entity recognition models
  '''
  maximum=0
#  for train_input,train_out in ner_read_TFRecord(sess,train_TFfileName,
#                                                 train_nerShapeFile,train_batch_size,args.epoch):
  for e in range(args.epoch):
    input_file_obj = open('data/WebQuestion/features/train_Data.txt')
    entMents = cPickle.load(open('data/WebQuestion/features/train_entMents.p','rb'))
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    #for ptr in xrange(0,len(train_input),args.batch_size):
    for train_input,train_out in get_input_figer_chunk_test_ner_train(args.batch_size,word2vecModel,300,input_file_obj,entMents,sentence_length=124):
    #for train_input,train_out in get_input_conll2003(word2vecModel,args.batch_size,300, open('data/conll2003/eng.train'), sentence_length=args.sentence_length):
      loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                         {model.input_data:WebQuestion_input,
                          model.output_data:WebQuestion_out,
                          model.num_examples:WebQuestion_num_example,
                          model.keep_prob:1})
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,WebQuestion_out,length)
      precision,recall,fscore = f1_chunk('CRF',args, pred, WebQuestion_out, length)
      

      m = fscore
      if m > maximum:
        maximum = m
        if maximum > 0.80:
          model.save(sess,args.restore,"WebQuestion") #optimize in the dev file!   
        params = {'length':length,'pred':pred,'target':WebQuestion_out,'args':args}
        cPickle.dump(params,open('data/WebQuestion/nerFeatures/WebQuestion_NERret.p','wb'))
        print("WebQuestion test: loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,accuracy,precision*100,recall*100,100*fscore))
        
      train_out = np.asarray(train_out,dtype=np.float32)
      train_out = np.argmax(train_out,2)
      num_example = np.shape(train_out)[0]
      _,loss1,tloss,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,totalLoss,model.length,model.output,model.unary_scores,model.transition_params],
                        {model.input_data:train_input,
                         model.output_data:train_out,
                         model.num_examples:num_example,
                         model.keep_prob:0.5})
      id_epoch += 1
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,train_out,length)
      precision,recall,fscore = f1_chunk('CRF',args, pred,train_out,length)
      if id_epoch %10==0:
        print("train: loss:%.4f total loss:%.4f accuracy:%f precision:%.2f recall:%.2f NER:%.2f" %(loss1,tloss,accuracy,precision*100,recall*100,100*fscore))
#  except:
#    print 'finished train'
if __name__=='__main__':
  tf.app.run()
