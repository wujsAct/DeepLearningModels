# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6

import math
from layers import Seq2SeqModel
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
import helpers

def make_seq2seq_model(**kwargs):
    args = dict(encoder_cell=LSTMCell(20),
                decoder_cell=LSTMCell(20),
                vocab_size=10,
                embedding_size=20,
                attention=False,
                bidirectional=False,
                debug=False)
    args.update(kwargs)
    return Seq2SeqModel(**args)


def train_on_copy_task(session, model,
                       length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=100,
                       max_batches=5000,
                       batches_in_epoch=1000,
                       verbose=True):

    batches = helpers.random_sequences(
        length_from=length_from, 
        length_to=length_to,
        vocab_lower=vocab_lower, 
        vocab_upper=vocab_upper,
        batch_size=batch_size)
    print batches
    loss_track = []
    try:
        for batch in range(max_batches+1):
            batch_data = next(batches)
#            print len(batch_data)
            fd = model.make_train_inputs(batch_data, batch_data)
#            t = session.run([model.decoder_logits_train], fd)
#            
#            print np.shape(t)
#           
#            PAD_SLICE = session.run([model.PAD_SLICE], fd)
#            print np.shape(PAD_SLICE)
#            
#            logits,targets,weight = session.run([model.logits,model.targets,model.loss_weights], fd)
#            print np.shape(logits),np.shape(targets),np.shape(weight)
#            print logits[0]
#            print targets[0]
#            print weight[0]
            
#            
#            decoder_outputs_train1 = session.run([model.decoder_outputs_train.rnn_output], fd)
#            decoder_outputs_train2 = session.run([model.decoder_outputs_train.sample_id], fd)
#            print decoder_outputs_train1
#            print np.shape(decoder_outputs_train2)
#            exit(0)

#
#            loss_weights = session.run([model.loss_weights], fd)
#            print 'loss_weights:',np.shape(loss_weights)
          
            _, l = session.run([model.train_op, model.loss], fd)
            loss_track.append(l)
#            print l
#            exit(0)
            
            if verbose:
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[model.encoder_inputs].T,
                            session.run(model.decoder_prediction_train, fd).T
                        )):
                        print('  sample {}:'.format(i + 1))
                        print('    enc input           > {}'.format(e_in))
                        print('    dec train predicted > {}'.format(dt_pred))
                        if i >= 2:
                            break
                    print()
    except KeyboardInterrupt:
        print('training interrupted')

    return loss_track


if __name__ == '__main__':
    import sys

    if 'fw-debug' in sys.argv:
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True)) as session:
            model = make_seq2seq_model(debug=True)
            session.run(tf.global_variables_initializer())
            decoder_target = session.run(model.decoder_targets)
            print decoder_target
            
            loss_weights,EOS_SLICE,decoder_train_inputs = session.run([model.loss_weights,model.EOS_SLICE,model.decoder_train_inputs])
            print 'loss_weights:',np.shape(loss_weights)
            print 'EOS_SLICE:',EOS_SLICE
            print 'decoder_train_inputs',decoder_train_inputs
            
            logits,targets = session.run([model.logits,model.targets])
            print np.shape(logits),np.shape(targets)
            print logits[0]
            print targets[0]
    
            decoder_prediction_train,loss = session.run([model.decoder_prediction_train,model.loss])
            print decoder_prediction_train
            print np.shape(decoder_prediction_train)
            print 'loss:',logits
            
    elif 'fw-inf' in sys.argv:
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True)) as session:
            model = make_seq2seq_model()
            session.run(tf.global_variables_initializer())
            fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])
            inf_out = session.run(model.decoder_prediction_inference, fd)
            print(inf_out)

    elif 'train' in sys.argv:
        tracks = {}

        tf.reset_default_graph()

        with tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True)) as session:
            model = make_seq2seq_model(attention=False)
            session.run(tf.global_variables_initializer())
            loss_track = train_on_copy_task(session, model)
            #print loss_track
    else:
        tf.reset_default_graph()
        session = tf.InteractiveSession(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True))
        model = make_seq2seq_model(debug=False)
        session.run(tf.global_variables_initializer())

        fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])

        inf_out = session.run(model.decoder_prediction_inference, fd)