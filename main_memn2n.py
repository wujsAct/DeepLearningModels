import os
import pprint
import tensorflow as tf
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 256, "internal state dimension [256]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("mem_size", 1000, "memory size [100]")
flags.DEFINE_integer("batch_size", 32, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 25, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "cnn", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir):
        print(" [*] Creating checkpoint directory...")
        os.makedirs(FLAGS.checkpoint_dir)
    #train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    #valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    #test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)

    #idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    #FLAGS.nwords = len(word2idx)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS,batch_size=FLAGS.batch_size, checkpoint_dir=FLAGS.checkpoint_dir, forward_only=False)
        if not FLAGS.is_test:
            print sess
            model.train(sess)
        else:
            model.run(train_data, valid_data)

if __name__ == '__main__':
    tf.app.run()
