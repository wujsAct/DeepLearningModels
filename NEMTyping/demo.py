import numpy as np
import tensorflow as tf
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec
import cPickle
from tqdm import tqdm

#shapes = np.asarray([3,4,5],dtype=np.int64)
#indices = np.asarray([[0,0,1],[1,2,2]],dtype=np.int64)
#vals =  np.array([1,2], dtype=np.int64)
#
#images = tf.sparse_placeholder(tf.float32)
#sess = tf.InteractiveSession()
#tt = sess.run(tf.sparse_tensor_to_dense(images), feed_dict={images:tf.SparseTensorValue(indices,vals,shapes)})
#print tt
import gensim

# Load Google's pre-trained Word2Vec model.
#models = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
word2vec = cPickle.load(open('data/wordvec_model_100.p'))
models = word2vec.wvec_model
vocab_dict = models.vocab
vocab_list = vocab_dict.keys()
vocab2id = {vocab_list[i]:i for i in range(len(vocab_list))}
id2vocab = {i:vocab_list[i] for i in range(len(vocab_list))}

vocabs=[]
for i in tqdm(range(len(id2vocab))):
  vocab = id2vocab[i]
  #print vocab
  vocabs.append(models[vocab])

lentVocabs = len(vocab2id)
print lentVocabs
vocab2id['NIL'] = lentVocabs
vocabs.append(np.zeros((100,)))
cPickle.dump(np.asarray(vocabs),open('data/word2vec_100.p','wb'))
cPickle.dump(vocab2id,open('data/vocab2id.p','wb'))