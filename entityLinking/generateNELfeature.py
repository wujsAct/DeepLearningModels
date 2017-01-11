import tensorflow as tf
import numpy as np
import cPickle
import time
from entityRecog import nameEntityRecognition,args #get seqLSTM features
from sklearn.metrics import f1_score
from utils import nelInputUtils as inputUtils
from utils import getLinkingFeature
from tqdm import tqdm

def nel_d3array_TFRecord(sess,TFfileName,nerShapeFile):
  print 'start to load data...'
  start_time = time.time()
  trainUtils = inputUtils(args.rawword_dim,"train")
  train_input = trainUtils.emb; train_out = trainUtils.tag; train_entliking= trainUtils.ent_linking;
  train_ent_mention_index = train_entliking['ent_mention_index']; train_ent_mention_link_feature=train_entliking['ent_mention_link_feature'];
  train_ent_mention_tag = train_entliking['ent_mention_tag']; train_ent_relcoherent = trainUtils.ent_relcoherent
  train_ent_linking_type = trainUtils.ent_linking_type; train_ent_linking_candprob = trainUtils.ent_linking_candprob
  print 'load data cost:',time.time()-start_time
  
  
  nerInstance = nameEntityRecognition(sess)
  lstm_output = nerInstance.getEntityRecognition(train_input,train_out)
  
  ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(args,lstm_output,train_ent_mention_index,train_ent_mention_tag,
                                                  train_ent_relcoherent,train_ent_mention_link_feature,train_ent_linking_type,train_ent_linking_candprob,0,flag='train')
  lents = np.shape(ent_mention_linking_tag_list)[0]                                                
  '''
  #named entity linking features: 
  modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature  
  
  
  '''
  
  '''
  #shapes:
  self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
    self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,self.args.rawword_dim])
    self.candidate_ent_type_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,self.args.figer_type_num])
    self.candidate_ent_prob_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,3])
    self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[None,2*self.args.rnn_size,1])
    self.candidate_ent_coherent_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
  '''
  param_dict={'ent_mention_linking_tag':[lents,args.candidate_ent_num],
              'candidate_ent_linking_feature':[lents,args.candidate_ent_num,int(args.rawword_dim)],
              'candidate_ent_coherent_feature':[lents,args.candidate_ent_num],
              'candidate_ent_type_feature':[lents,args.candidate_ent_num,args.figer_type_num],
              'candidate_ent_prob_feature':[lents,args.candidate_ent_num,3],
              'ent_mention_lstm_feature':[lents,2*args.rnn_size,1]
              }
              
  cPickle.dump(param_dict,open(nerShapeFile,'wb'))
  
  writer = tf.python_io.TFRecordWriter(TFfileName)
  for i in tqdm(xrange(lents)):
    lab = ent_mention_linking_tag_list[i]
    feat1 = candidate_ent_relcoherent_feature[i]
    feat2 = np.reshape(candidate_ent_linking_feature[i],[args.candidate_ent_num*int(args.rawword_dim)])
    feat3 = np.reshape(candidate_ent_type_feature[i],[args.candidate_ent_num*args.figer_type_num])
    feat4 = np.reshape(candidate_ent_prob_feature[i],[args.candidate_ent_num*3])
    feat5 = np.reshape(ent_mention_lstm_feature[i],[2*args.rnn_size])
    
    example = tf.train.Example(features=tf.train.Features(feature={
     "ent_mention_linking_tag":tf.train.Feature(int64_list=tf.train.Int64List(value=lab)),
     "candidate_ent_coherent_feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat1)),
     "candidate_ent_linking_feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat2)),
     "candidate_ent_type_feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat3)),
     "candidate_ent_prob_feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat4)),
     "ent_mention_lstm_feature":tf.train.Feature(float_list=tf.train.FloatList(value=feat5))     
    }))
    writer.write(example.SerializeToString())
  writer.close()
  

if __name__=="__main__":
#  nerfile = "ner.tfrecords"
#  nershape = "ner.shape"
  #mu = 0;sigma=1
  #rarray=np.random.normal(mu,sigma,[2580,2,5])
  #label=np.zeros([2580,2,3],dtype=np.int64)
  #ner_d3array_TFRecord(rarray,label,nerfile,nershape)
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  TFfileName = 'data/aida/trainNEL.tfrecord'
  nerShapeFile ='data/aida/trainNEL.shape'
  nel_d3array_TFRecord(sess,TFfileName,nerShapeFile)