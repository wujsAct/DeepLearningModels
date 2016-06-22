import os
import math
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from base_model import Model
from data_utils import QADataset
import time

class MemN2N(Model):
    def __init__(self,config,batch_size=32, checkpoint_dir="checkpoint_dir", forward_only=False,query_n_input=50,context_n_input=1000,data_dir="data", dataset_name="cnn"):
        super(MemN2N, self).__init__()
        print 'start to import'
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        #self.mem_size = config.mem_size
        self.mem_size = context_n_input
        self.q_men_size = query_n_input
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir
        self.log_loss = []
        self.log_perp = []
        self.data_dir= data_dir
        self.dataset_name = dataset_name
        self.config = config

    def build_memory(self):
        vocab_fname = os.path.join(self.data_dir,self.dataset_name,self.dataset_name+'.vocab')
        qaData_t = QADataset(self.data_dir,self.dataset_name,vocab_fname)
        if not self.vocab:
            self.vocab,self.n_entities = qaData_t.initialize_vocabulary()
            print(" [*] Loading vocab finished.")

        self.nwords = len(self.vocab)
        
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.context_time = tf.placeholder(tf.int32, [None, self.mem_size], name="context_time")
        self.q_time = tf.placeholder(tf.int32,[None,self.q_men_size],name='q_time')
        self.target = tf.placeholder(tf.int64, [self.batch_size], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")
        self.context_seq_length = tf.placeholder(tf.int32,[self.batch_size])
        
        self.query = tf.placeholder(tf.int32, [self.batch_size, self.q_men_size], name="query")
        self.query_seq_length= tf.placeholder(tf.int32,[self.batch_size])
        self.hid = []
        #self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = self.config.init_lr
        self.loss = None
        self.step = None
        self.optim = None
        
        self.global_step = tf.Variable(0, name="global_step")
        
        #for context
        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
       
        #for query
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        
        #self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))
        
        
        # Temporal Encoding for context
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_C = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
          
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.context_time)
        
        temp1 = tf.constant([[0]*(self.edim)]*self.mem_size,dtype=tf.float32)
        Ain_c = tf.pack([tf.concat(0,[tf.slice(Ain_c,[idx,0,0],[1,lent,self.edim])[0,:,:],tf.slice(temp1,[0,0],[self.mem_size-lent,self.edim])])for idx,lent in enumerate(tf.unpack(self.context_seq_length))])
        
        Ain = tf.add(Ain_c, Ain_t)
        
        Cin_c = tf.nn.embedding_lookup(self.C, self.context)
        Cin_t = tf.nn.embedding_lookup(self.T_C, self.context_time)
        Cin = tf.add(Cin_c, Cin_t)
        
        
        # m_i = sum A_ij * x_ij + T_A_i
        
        #Temporal Encoding for query
        self.T_B = tf.Variable(tf.random_normal([self.q_men_size, self.edim], stddev=self.init_std))
        
        Bin_q = tf.nn.embedding_lookup(self.B,self.query)
        Bin_t = tf.nn.embedding_lookup(self.T_B,self.q_time)
        temp2 = tf.constant([[0]*(self.edim)]*self.q_men_size,dtype=tf.float32)
        Bin_q = tf.pack([tf.concat(0,[tf.slice(Bin_q,[idx,0,0],[1,lent,self.edim])[0,:,:],tf.slice(temp2,[0,0],[self.q_men_size-lent,self.edim])])for idx,lent in enumerate(tf.unpack(self.query_seq_length))])
        Bin = tf.add(Bin_q, Bin_t)
        #get u: need to sum  the Bin to get u
        u = [tf.reduce_sum(Bin[i,:,:],0) for i in range(self.batch_size)]
        #need to truncation: because of the different length of query! 
        self.hid.append(u)
        
        for h in xrange(self.nhop):
            #self.hid[-1] shape is: [self.batch_size,1,self.edim]; while Ain shape is [batch_size,mem_size,edim]
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.batch_matmul(self.hid3dim, Ain, adj_y=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)
            
            #get o
            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Cout = tf.batch_matmul(probs3dim, Cin)
            o = tf.reshape(Cout, [-1,self.edim])
             
            #get o + u
            print self.hid[-1]
            print o
            Dout = tf.add(self.hid[-1], o)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(1, [F, K]))

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.n_entities],dtype=tf.float32))
        self.y_ = tf.matmul(self.hid[-1], self.W)
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_, self.target)
        correct_prediction = tf.equal(self.target, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def train(self,sess,epoch=25,learning_rate=0.0002):
        start_time = time.time()
        self.build_model()
        self.lr = tf.Variable(self.current_lr)
        sess.run(tf.initialize_all_variables())
        start = time.clock()
        self.opt = tf.train.GradientDescentOptimizer(self.lr)
        #to solve gradient clip problem!
        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]
        
        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)
            
        print " [*] Calculating gradient and loss finished. Take %.2fs"  %(time.clock() - start)
            
        if self.load(sess, self.checkpoint_dir, self.dataset_name):
            print(" [*] MemoryN2N checkpoint is loaded.")
        else:
            print(" [*] There is no checkpoint for this model.")
            
        cost = 0
        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        context_time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        q_time = np.ndarray([self.batch_size,self.q_men_size],dtype=np.int32)
        #target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])
        
        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            context_time[:,t].fill(t)
        for t in xrange(self.q_men_size):
            q_time[:,t].fill(t)
        

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)
        counter = 0
        data_max_idx=280096
        for epoch_idx in tqdm(xrange(self.nepoch)):
            vocab_fname = os.path.join(self.data_dir,self.dataset_name,self.dataset_name+'.vocab')
            qaData = QADataset(self.data_dir,self.dataset_name,vocab_fname)
            data_loader = qaData.load_dataset2(self.batch_size)
            
            while True:
                try:
                    context,ctx_seq_length,query,q_seq_length,target = data_loader.next()
                    _, cost, accuracy,self.step = sess.run([self.optim,
                                                        self.loss,
                                                        self.accuracy,
                                                        self.global_step],
                                                        feed_dict={
                                                            self.input: x,
                                                            self.context_time: context_time,
                                                            self.q_time: q_time,
                                                            self.target: target,
                                                            self.query:query,
                                                            self.context:context,
                                                            self.context_seq_length:ctx_seq_length,
                                                            self.query_seq_length:q_seq_length
                                                            })
                    #cost += np.sum(loss)
                    if counter % 10 == 0:
                        #writer.add_summary(summary_str, counter)
                        data_idx = (counter+1) * self.batch_size
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                           % (epoch_idx, data_idx, data_max_idx, time.time() - start_time, np.mean(cost), accuracy))
                    counter += 1
                except StopIteration:
                    break        
            self.save(sess, self.checkpoint_dir, self.dataset_name)   #save the model
            