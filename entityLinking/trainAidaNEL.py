# -*- coding: utf-8 -*-
'''
@editor: wujs
function: we add entity linking module
revise: 2017/1/8
'''

import tensorflow as tf
import time
from model import ctxSum
from entityRecog import nameEntityRecognition,pp,flags,args,f1 #get seqLSTM features
from sklearn.metrics import f1_score
from utils import nelInputUtils as inputUtils
from utils import getLinkingFeature
import numpy  as np

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  print 'start to load data...'
  start_time = time.time()
  testaUtils = inputUtils(args.rawword_dim,"testa")
  testa_input = testaUtils.emb; testa_out = testaUtils.tag; testa_entliking= testaUtils.ent_linking;
  testa_ent_mention_index = testa_entliking['ent_mention_index']; testa_ent_mention_link_feature=testa_entliking['ent_mention_link_feature'];
  testa_ent_mention_tag = testa_entliking['ent_mention_tag']; testa_ent_relcoherent = testaUtils.ent_relcoherent
  testa_ent_linking_type = testaUtils.ent_linking_type; testa_ent_linking_candprob = testaUtils.ent_linking_candprob

  trainUtils = inputUtils(args.rawword_dim,"train")
  train_input = trainUtils.emb; train_out = trainUtils.tag; train_entliking= trainUtils.ent_linking;
  train_ent_mention_index = train_entliking['ent_mention_index']; train_ent_mention_link_feature=train_entliking['ent_mention_link_feature'];
  train_ent_mention_tag = train_entliking['ent_mention_tag']; train_ent_relcoherent = trainUtils.ent_relcoherent
  train_ent_linking_type = trainUtils.ent_linking_type; train_ent_linking_candprob = trainUtils.ent_linking_candprob

  testbUtils = inputUtils(args.rawword_dim,"testb")
  testb_input = testbUtils.emb; testb_out = testbUtils.tag; testb_entliking= testbUtils.ent_linking
  testb_ent_mention_index = testb_entliking['ent_mention_index']; testb_ent_mention_link_feature=testb_entliking['ent_mention_link_feature'];
  testb_ent_mention_tag = testb_entliking['ent_mention_tag']; testb_ent_relcoherent = testbUtils.ent_relcoherent
  testb_ent_linking_type = testbUtils.ent_linking_type; testb_ent_linking_candprob = testbUtils.ent_linking_candprob
  print 'cost:', time.time()-start_time,' to load data'

  print 'start to initialize parameters'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:
    nerInstance = nameEntityRecognition(sess)   #load named entity recoginition models
    lstm_output_train = nerInstance.getEntityRecognition(train_input,train_out)
    lstm_output_testa = nerInstance.getEntityRecognition(testa_input,testa_out)
    lstm_output_testb = nerInstance.getEntityRecognition(testb_input,testb_out)
    modelNEL = ctxSum(args)  #build named entity linking models
    loss_linking = modelNEL.linking_loss
    optimizer = tf.train.AdamOptimizer(0.05)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ctxSum')
    print 'tvars_linking:',tvars
    grads,_ = tf.clip_by_global_norm(tf.gradients(loss_linking,tvars),5)
    train_op_linking = optimizer.apply_gradients(zip(grads,tvars))
    sess.run(tf.initialize_all_variables())

    if modelNEL.load(sess,args.restore,"aida"):
      print "[*] ctxSum is loaded..."
    else:
      print "[*] There is no checkpoint for aida"

    id_epoch = 0
    '''
    @train named entity linking models
    '''
    maximum_linking=0
    for e in range(args.epoch):
      id_epoch = 0
      print 'Epoch: %d------------' %(e)
      average_linking_accuracy_train = 0;average_loss_train = 0
      for ptr in xrange(0,len(train_input),args.batch_size):
        id_epoch = id_epoch + 1
        lstm_output = lstm_output_train[ptr:min(ptr+args.batch_size,len(train_input))];

        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(args,lstm_output,train_ent_mention_index,train_ent_mention_tag,\
                                                  train_ent_relcoherent,train_ent_mention_link_feature,train_ent_linking_type,train_ent_linking_candprob,ptr,flag='train')
        _,loss2,accuracy,pred = sess.run([train_op_linking,loss_linking,modelNEL.accuracy,modelNEL.prediction],
                                 {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
        average_linking_accuracy_train += accuracy
        average_loss_train += loss2
        if id_epoch%20==0:
          ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(args,lstm_output_testa,testa_ent_mention_index,testa_ent_mention_tag,\
                                                  testa_ent_relcoherent,testa_ent_mention_link_feature,testa_ent_linking_type,testa_ent_linking_candprob,0,flag='testa')
          loss2,accuracy,pred = sess.run([loss_linking,modelNEL.accuracy,modelNEL.prediction],
                                 {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
          f1_micro,f1_macro = f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),\
                                          f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print 'testa total loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro


          if accuracy > maximum_linking:
            maximum_linking = accuracy
            modelNEL.save(sess,args.restore,"aida")
            ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature=\
                                                  getLinkingFeature(args,lstm_output_testb,testb_ent_mention_index,testb_ent_mention_tag,\
                                                  testb_ent_relcoherent,testb_ent_mention_link_feature,testb_ent_linking_type,testb_ent_linking_candprob,0,flag='testb')
            fscore = f1(args, pred, testb_out, length)
            loss2,accuracy,pred = sess.run([loss_linking,modelNEL.accuracy,modelNEL.prediction],
                                 {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
            f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),\
                                          f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
            print 'testb total loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
            print "-----------------"
      print 'average linking accuracy:',average_linking_accuracy_train/id_epoch, average_loss_train/id_epoch



if __name__=="__main__":
  tf.app.run()
  #定义linking model
