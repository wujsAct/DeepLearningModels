import os
import numpy as np
import tensorflow as tf
from model import ctxCNN,seqLSTM
from utils import inputUtils
import pprint
import random
import time
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",64,"batch size of training")
flags.DEFINE_integer("sentence_length",50,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",111,"hidden dimension of rnn")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.003,"apply dropout during training")

args = flags.FLAGS

def f1(args, prediction, target, length):
  tp = np.array([0] * (args.class_size + 1))
  fp = np.array([0] * (args.class_size + 1))
  fn = np.array([0] * (args.class_size + 1))
  target = np.argmax(target, 2)
  prediction = np.argmax(prediction, 2)
  for i in range(len(target)):
    for j in range(length[i]):
      if target[i, j] == prediction[i, j]:
        tp[target[i, j]] += 1
      else:
        fp[target[i, j]] += 1
        fn[prediction[i, j]] += 1
  unnamed_entity = args.class_size - 1
  for i in range(args.class_size):
    if i != unnamed_entity:
      tp[args.class_size] += tp[i]
      fp[args.class_size] += fp[i]
      fn[args.class_size] += fn[i]
  precision = []
  recall = []
  fscore = []
  for i in range(args.class_size + 1):
    precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
    recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
    fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
  
  return fscore


def main(_):
  pp.pprint(flags.FLAGS.__flags)
  if not os.path.exists(args.restore):
    print("[*] creating checkpoint directory...")
    os.makedirs(args.restore)
  '''
  @function: load the train and test datasets
  '''
  print 'start to load data...'
  start = time.time()
  trainUtils = inputUtils(args.rawword_dim,"train")
  train_input = trainUtils.emb; train_out = trainUtils.tag
  
  testaUtils = inputUtils(args.rawword_dim,"testa")
  testa_input = testaUtils.emb; testa_out = testaUtils.tag
  
  testbUtils = inputUtils(args.rawword_dim,"testb")
  testb_input = testbUtils.emb; testb_out = testbUtils.tag
  print 'cost:', time.time()-start,' to load data'
  
  '''
  @function: initialize the parameters and start the interactiveSession.
  '''
  model = seqLSTM(args)
  loss = model.loss
  
  start_time = time.time()
  print 'gradient computing...'
  optimizer = tf.train.AdamOptimizer(args.learning_rate)
  tvars = tf.trainable_variables()
  grads,_ = tf.clip_by_global_norm(tf.gradients(loss,tvars),10)
  train_op = optimizer.apply_gradients(zip(grads,tvars))
  print 'gradient computing cost time:', time.time()-start_time
  
  print 'start to initialize parameters'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=16,inter_op_parallelism_threads=16)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  print 'initiliaze parameters cost time:', time.time()-start
  init = tf.initialize_all_variables()
  sess.run(init)
  
  maximum = 0
  saver = tf.train.Saver()  #prepare to save the trained models
  start_tr = time.time()
  id_epoch = 0
  for e in range(args.epoch):
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    for ptr in xrange(0,len(train_input),args.batch_size):
      id_epoch = id_epoch + 1
      sess.run(train_op, {model.input_data:train_input[ptr:ptr+args.batch_size],
                          model.output_data:train_out[ptr:ptr+args.batch_size],
                          model.keep_prob:0.5})
      if id_epoch%20==0:
        loss1,pred,length = sess.run([loss,model.prediction,model.length],
                          {model.input_data:train_input[ptr:ptr+args.batch_size],
                          model.output_data:train_out[ptr:ptr+args.batch_size],
                          model.keep_prob:0.5})
        fscore = f1(args, pred, train_out[ptr:ptr+args.batch_size], length)
        print("train: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        
                    
        loss1,pred,length = sess.run([loss,model.prediction,model.length],
                             {model.input_data:testa_input,
                              model.output_data:testa_out,
                              model.keep_prob:1})
        fscore = f1(args, pred, testa_out, length)
        print "-----------------"
        print("testa: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        m = fscore[args.class_size]
        if m > maximum:
          maximum = m
          loss1,pred,length = sess.run([loss,model.prediction,model.length],
                             {model.input_data:testb_input,
                              model.output_data:testb_out,
                              model.keep_prob:1})
          fscore = f1(args, pred, testb_out, length)
          print("testb: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        print "-----------------" 
if __name__ == '__main__':
  tf.app.run()      
