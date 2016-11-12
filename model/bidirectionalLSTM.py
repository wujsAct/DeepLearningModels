import time
import tensorflow as tf
from data_utils import QADataset
from base import Model
import os
import Utils  #for xavier_init (it is a trick)
import numpy as np
import pickle
from tensorflow.contrib.layers.python.layers import batch_norm

#this version we delete the merge sum!
try:
  linear = tf.nn.rnn_cell.linear
except:
  from tensorflow.python.ops.rnn_cell import _linear as linear
class BiLSTM(Model):
  """Neural Answer Selection Model"""

  def __init__(self,data_dir,dataset,
               batch_size=100,embed_dim=100,
               query_n_input=30,context_n_input=800,
               h_dim=300, learning_rate=0.2, epoch=300,
               decay_rate=0.96,
               checkpoint_dir="checkpoint",transfer_fct=tf.nn.relu,drop_out=True):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and
       test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.drop_out = drop_out
    self.transfer_fct = transfer_fct
    self.embed_dim = embed_dim
    self.epoch = epoch
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir
    self.q_output_size = embed_dim*2
    self.ctx_output_size = embed_dim*2
    self.data_dir = data_dir
    self.dataset_name = dataset
    self.dataset = dataset
    self.query_n_input = query_n_input
    self.ctx_n_input = context_n_input
    self.vocab = None
    self.n_entities = 584
    self._attrs=["batch_size", "embed_dim","learning_rate"]
    self.step = tf.Variable(0, trainable=False)  
    self.lr = tf.train.exponential_decay(
        learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")
    self.build_model()
    

  def build_model(self):
    vocab_fname = os.path.join(self.data_dir,self.dataset_name,'new_'+self.dataset_name+'.vocab')
    qaData_t = QADataset(self.data_dir,self.dataset_name,vocab_fname,self.query_n_input,self.ctx_n_input)
    if not self.vocab:
      self.vocab,self.n_entities = qaData_t.initialize_vocabulary()
      print(" [*] Loading vocab finished.")
    
    self.vocab_size = len(self.vocab)
    print self.vocab_size

    vocab_vector = pickle.load(open(self.data_dir+'/'+self.dataset_name+'/new_vocab.word2vec','rb'))
    with tf.device('/gpu:0'):
      #self.emb = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_dim),0,1,dtype=tf.float32), name='emb')
      self.emb = tf.Variable(vocab_vector,name='emb',dtype=tf.float32)
      self.ctx_LSTM_inputs =tf.placeholder(tf.int32,[self.batch_size,self.ctx_n_input])
      self.ctx_inputs_sequence = tf.placeholder(tf.int32,[self.batch_size])
        
      self.embed_inputs_ctx = tf.nn.embedding_lookup(self.emb, tf.transpose(self.ctx_LSTM_inputs))
      
      self.query_LSTM_inputs = tf.placeholder(tf.int32,[self.batch_size,self.query_n_input])
      self.query_inputs_sequence = tf.placeholder(tf.int32,[self.batch_size])    
      self.embed_inputs_q = tf.nn.embedding_lookup(self.emb, tf.transpose(self.query_LSTM_inputs))  
      
      self.keep_prob = tf.placeholder(tf.float32)
      
      self.build_biLSTM()
      
      self.y_loss = tf.nn.softmax_cross_entropy_with_logits(self.y_,self.y)
      correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print 'self.accuracy',self.accuracy
      #tf.scalar_summary("accuracy", self.accuracy)
    
      self.predict_loss = tf.reduce_mean(self.y_loss)
  
      self.loss = self.predict_loss
      
      print 'predict_loss',self.predict_loss
      
      print 'start to optimizer the function'
      time1 = time.time()
      self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
      
      print 'compile gradient time', time.time()-time1
    
  #@2106/11/3 need to revise the code
  def build_biLSTM(self):
    layers=1
    with tf.device('/gpu:0'):
#      initializer = tf.random_normal_initializer(0, 0.1)
      lstm_fw_cell_q = tf.nn.rnn_cell.GRUCell(self.embed_dim)
      lstm_bw_cell_q = tf.nn.rnn_cell.GRUCell(self.embed_dim)
#      print 'lstm_fw_cell_q.state_size:',lstm_fw_cell_q.state_size
      
      if self.drop_out:
          lstm_fw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_q, output_keep_prob=self.keep_prob)
          lstm_bw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_q, output_keep_prob=self.keep_prob)
          
      lstm_fw_multicell_q = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell_q]*layers)
      lstm_bw_multicell_q = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell_q]*layers)
      
#      initial_state_fw_q=  lstm_fw_multicell_q.zero_state(self.batch_size, tf.float32)
#      initial_state_bw_q= lstm_bw_multicell_q.zero_state(self.batch_size,tf.float32)
      
      q_out,_,_ = tf.nn.bidirectional_rnn(lstm_fw_multicell_q,lstm_bw_multicell_q,
                                      tf.unpack(self.embed_inputs_q),
                                      dtype=tf.float32,
                                      sequence_length=self.query_inputs_sequence,scope="query_birnn") 
      '''
      'outputs' is a list of output at every timestep, we pack them in a Tensor
      and change back dimension to [batch_size, n_step, n_input]
      '''
      q_out = tf.pack(q_out)
      q_out = tf.transpose(q_out, [1, 0, 2])
      print q_out
      #Hack to build the indexing and retrieve the right output.
      # Start indices for each sample
      index = tf.range(0, self.batch_size) * self.query_n_input + (self.query_inputs_sequence - 1)
      # Indexing
      self.q_out_original = tf.nn.l2_normalize(tf.gather(tf.reshape(q_out, [-1, self.q_output_size]), index),1)
          
      print 'q_out_original',self.q_out_original
      
      self.q_mask = tf.placeholder(tf.float32,[self.batch_size,self.query_n_input,self.q_output_size])
      self.q_out_real = tf.nn.l2_normalize(tf.mul(q_out,self.q_mask),1)
      
      time_answer = time.time()
      print 'start to answer lstm'
      
    with tf.device('/gpu:0'):
#      initializer = tf.random_normal_initializer(0,0.1)
      lstm_fw_cell_ctx = tf.nn.rnn_cell.GRUCell(self.embed_dim)
      lstm_bw_cell_ctx = tf.nn.rnn_cell.GRUCell(self.embed_dim)
      
      if self.drop_out:
          lstm_fw_cell_ctx = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_ctx, output_keep_prob=self.keep_prob)
          lstm_bw_cell_ctx = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_ctx, output_keep_prob=self.keep_prob)
          
      lstm_fw_multicell_ctx = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell_ctx]*layers)
      lstm_bw_multicell_ctx = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell_ctx]*layers)
      
#      initial_state_fw_ctx=  lstm_fw_multicell_ctx.zero_state(self.batch_size, tf.float32)
#      initial_state_bw_ctx= lstm_bw_multicell_ctx.zero_state(self.batch_size,tf.float32)
      
      print 'answer intialize time:',time.time()-time_answer 
      a_out,_,_ = tf.nn.bidirectional_rnn(lstm_fw_multicell_ctx,lstm_bw_multicell_ctx,
                                          tf.unpack(self.embed_inputs_ctx),
                                          
                                          dtype=tf.float32,
                                          sequence_length=self.ctx_inputs_sequence,scope="ctx_birnn") 
                                                       
      a_out=tf.pack(a_out)
        
      print 'a_out',a_out
      print 'answer embedding time:',time.time()-time_answer
          
      self.a_out = tf.transpose(a_out, [1, 0, 2])
      # Start indices for each sample
      index = tf.range(0, self.batch_size) * self.ctx_n_input + (self.ctx_inputs_sequence - 1)
      # Indexing
      self.a_out_last = tf.nn.l2_normalize(tf.gather(tf.reshape(self.a_out, [-1, self.ctx_output_size]), index),1)
      print 'self.a_out_last',self.a_out_last
        
      self.y = tf.placeholder(tf.float32,[self.batch_size,self.n_entities])
      
      self.a_mask = tf.placeholder(tf.float32,[self.batch_size,self.ctx_n_input,self.ctx_output_size])
      self.a_out_real = tf.nn.l2_normalize(tf.mul(self.a_out,self.a_mask),1)
      print 'self.a_out_real',self.a_out_real
     
     
      weights_attention ={
                'W': tf.Variable(Utils.xavier_init(self.ctx_output_size,self.q_output_size))
                }
          
      reshape_a_out_real= tf.reshape(self.a_out_real,[self.batch_size * self.ctx_n_input,self.ctx_output_size])
      print 'reshape_a_out_real:',reshape_a_out_real
      Wa_w = tf.reshape(tf.matmul(reshape_a_out_real,weights_attention['W']),[self.batch_size,self.ctx_n_input,self.ctx_output_size])
      q_out_original_reshape = tf.expand_dims(self.q_out_original,2)
      
      fai = tf.batch_matmul(Wa_w,q_out_original_reshape)
      print 'fai:',fai
      fai = tf.nn.softmax(tf.reshape(fai,[self.batch_size,self.ctx_n_input]))
      print 'reshape fai:',fai
      fai = tf.expand_dims(fai,2)
      print 'expand reshape fai:',fai
          
      print 'self.a_out_real:',self.a_out_real
      self.final_feature = tf.nn.l2_normalize(tf.reduce_sum(tf.mul(self.a_out_real,fai),1),1)

      #self.final_feature = self.z_a_h
      weights_y ={
        'w_l1':tf.Variable(Utils.xavier_init(self.ctx_output_size,self.ctx_output_size)),
        'w_l2':tf.Variable(Utils.xavier_init(self.ctx_output_size,self.n_entities))
      }
      bfinal= tf.Variable(tf.zeros([self.n_entities], dtype=tf.float32))
      #compare those two different dropout situations!
      w_l1 = tf.nn.relu(batch_norm(tf.matmul(self.final_feature,weights_y['w_l1'])))
      
      if self.drop_out:
        w_l1 = tf.nn.dropout(w_l1,self.keep_prob)
      self.y_ = tf.matmul(w_l1,weights_y['w_l2'])+bfinal
     
      
  def train(self,config,sess):
    with tf.device('/gpu:0'):
      start_time = time.time()
      self.sess = sess
      #merged_sum = tf.merge_all_summaries()
      #writer = tf.train.SummaryWriter("./logs", self.sess.graph)
      tf.initialize_all_variables().run()
      self.load(self.checkpoint_dir)
      #sess.graph.finalize()
      counter = 0
      print 'initialize time:',time.time()-start_time
      data_max_idx=380298
      start_traing_time = time.time()
      vocab_fname1 = os.path.join(self.data_dir,self.dataset_name,'new_'+self.dataset_name+'.vocab')
      qaData1 = QADataset(self.data_dir,self.dataset_name,vocab_fname1,self.query_n_input,self.ctx_n_input)
      for epoch_idx in xrange(self.epoch):
            
        data_loader = qaData1.load_dataset2(self.batch_size,'training')
        idx =0
        
        while True:
          try:
            '''
            @training data for every iterations!
            '''    
            ctx,ctx_seq_length,q,q_seq_length,y = data_loader.next()
            ctx_mask=[]
            q_mask=[]
            for i in xrange(self.batch_size):
              ctx_mask.append([[1]*self.embed_dim*2]*min(ctx_seq_length[i],self.ctx_n_input) +[[0.0]*self.embed_dim*2]*(self.ctx_n_input-ctx_seq_length[i]))
              q_mask.append([[1]*self.embed_dim*2]*min(q_seq_length[i],self.query_n_input) +[[0.0]*self.embed_dim*2]*(self.query_n_input-q_seq_length[i]))
              ctx_seq_length[i] = min(ctx_seq_length[i],self.ctx_n_input)
              q_seq_length[i] = min(q_seq_length[i],self.query_n_input)
              
            ctx_mask = np.asarray(ctx_mask)
            q_mask= np.asarray(q_mask)
            _,loss,predict_loss,accuracy = self.sess.run([self.optim, self.loss, self.predict_loss,self.accuracy], 
                                                    feed_dict={self.ctx_LSTM_inputs:ctx,
                                                       self.ctx_inputs_sequence:ctx_seq_length,
                                                       self.query_LSTM_inputs:q,
                                                       self.query_inputs_sequence:q_seq_length,
                                                       self.y:y,
                                                       self.keep_prob:0.8,
                                                       self.a_mask:ctx_mask,self.q_mask:q_mask})
                                                       
            if idx % 1280 == 0:
              print("Epoch: [%2d] [%4d/%4d] time: %.4f,loss: %.8f,predict_loss:%.8f,accuracy:%.6f" \
                % (epoch_idx,idx ,data_max_idx,time.time()-start_time,loss,predict_loss,accuracy))
            idx += self.batch_size
          except StopIteration:
            print 'start to save the checkpoint'
            break
        '''
        #validation for every iteration!
        '''
        data_loader_validation = qaData1.load_dataset2(self.batch_size,'validation')
        validate_loss = []
        validate_accuracy =[]
        while True:
          try:
            ctx,ctx_seq_length,q,q_seq_length,y = data_loader_validation.next()
            ctx_mask=[]
            q_mask=[]
            for i in xrange(self.batch_size):
              ctx_mask.append([[1]*self.embed_dim*2]*min(ctx_seq_length[i],self.ctx_n_input) +[[0.0]*self.embed_dim*2]*(self.ctx_n_input-ctx_seq_length[i]))
              q_mask.append([[1]*self.embed_dim*2]*min(q_seq_length[i],self.query_n_input) +[[0.0]*self.embed_dim*2]*(self.query_n_input-q_seq_length[i]))
              ctx_seq_length[i] = min(ctx_seq_length[i],self.ctx_n_input)
              q_seq_length[i] = min(q_seq_length[i],self.query_n_input)
              
            ctx_mask = np.asarray(ctx_mask)
            q_mask= np.asarray(q_mask)
            loss,accuracy = self.sess.run([self.loss,self.accuracy], feed_dict={self.ctx_LSTM_inputs:ctx,
                                                                                 self.ctx_inputs_sequence:ctx_seq_length,
                                                                                 self.query_LSTM_inputs:q,
                                                                                 self.query_inputs_sequence:q_seq_length,
                                                                                 self.y:y,
                                                                                 self.keep_prob:1,
                                                                                 self.a_mask:ctx_mask,self.q_mask:q_mask})
            validate_loss.append(loss)
            validate_accuracy.append(accuracy)
          except StopIteration:
            print 'print validation ...'
            print("validation loss:%.8f,validation accuracy:%.8f" %(np.average(validate_loss),np.average(validate_accuracy)))
            break
        '''
        @test data for every iterations!
        '''      
        data_loader_test = qaData1.load_dataset2(self.batch_size,'test')
        test_loss = []
        test_accuracy =[]
        while True:
          try:
            ctx,ctx_seq_length,q,q_seq_length,y = data_loader_test.next()
            ctx_mask=[]
            q_mask=[]
            for i in xrange(self.batch_size):
              ctx_mask.append([[1]*self.embed_dim*2]*min(ctx_seq_length[i],self.ctx_n_input) +[[0.0]*self.embed_dim*2]*(self.ctx_n_input-ctx_seq_length[i]))
              q_mask.append([[1]*self.embed_dim*2]*min(q_seq_length[i],self.query_n_input) +[[0.0]*self.embed_dim*2]*(self.query_n_input-q_seq_length[i]))
              ctx_seq_length[i] = min(ctx_seq_length[i],self.ctx_n_input)
              q_seq_length[i] = min(q_seq_length[i],self.query_n_input)
            ctx_mask = np.asarray(ctx_mask)
            q_mask= np.asarray(q_mask)
            loss,accuracy = self.sess.run([self.loss,self.accuracy], feed_dict={self.ctx_LSTM_inputs:ctx,
                                                                                 self.ctx_inputs_sequence:ctx_seq_length,
                                                                                 self.query_LSTM_inputs:q,
                                                                                 self.query_inputs_sequence:q_seq_length,
                                                                                 self.y:y,
                                                                                 self.keep_prob:1,
                                                                                 self.a_mask:ctx_mask,self.q_mask:q_mask})
            test_loss.append(loss)
            test_accuracy.append(accuracy)
          except StopIteration:
            print 'print test ...'
            print("test_loss:%.8f, test_accuracy:%.6f" %(np.average(test_loss),np.average(test_accuracy)))
            break
      self.save(self.checkpoint_dir)  
