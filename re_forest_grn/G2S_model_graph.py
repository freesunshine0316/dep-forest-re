import tensorflow as tf
import encoder_utils
import graph_encoder_utils
import padding_utils
from tensorflow.python.ops import variable_scope
import numpy as np
import random


def _clip_and_normalize(word_probs, epsilon):
    '''
    word_probs: 1D tensor of [vsize]
    '''
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True) # scale probs to sum to 1


def range_mask(sentence_size_max, starts, ends):
    mask1 = tf.cast(tf.sequence_mask(starts, maxlen=sentence_size_max),
            dtype=tf.float32)
    mask2 = tf.cast(tf.sequence_mask(ends, maxlen=sentence_size_max),
            dtype=tf.float32)
    return mask2 - mask1


def range_repres(final_repres, sentence_size_max, starts, ends):
    mask = range_mask(sentence_size_max, starts, ends) # [batch, sentence_size_max]
    return tf.multiply(final_repres, tf.expand_dims(mask, axis=-1))


def collect_by_indices(final_repres, indices):
    batch_size = tf.shape(indices)[0]
    entity_num = tf.shape(indices)[1]
    entity_size = tf.shape(indices)[2]
    idxs = tf.range(0, limit=batch_size) # [batch]
    idxs = tf.reshape(idxs, [-1, 1, 1]) # [batch, 1, 1]
    idxs = tf.tile(idxs, [1, entity_num, entity_size])
    indices = tf.maximum(indices, tf.zeros_like(indices, dtype=tf.int32))
    indices = tf.stack((idxs,indices), axis=3) # [batch,2,indices,2]
    return tf.gather_nd(final_repres, indices)


class ModelGraph(object):
    def __init__(self, word_vocab, char_vocab, pos_vocab, edgelabel_vocab, options, mode='train'):
        # the value of 'mode' can be:
        #  'train',
        #  'evaluate'
        self.mode = mode

        # is_training controls whether to use dropout
        is_training = True if mode in ('train', ) else False

        self.options = options
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab

        # sequential encoder that can take 0 LSTM layers
        self.encoder = encoder_utils.SeqEncoder(options,
                word_vocab, char_vocab, pos_vocab)
        word_repres, word_dim, sentence_repres, sentence_dim, seq_mask = \
                self.encoder.encode(is_training=is_training)

        # encode the input instance
        # encoder.graph_hidden [batch, node_num, vsize]
        # encoder.graph_cell [batch, node_num, vsize]
        self.graph_encoder = graph_encoder_utils.GraphEncoder(options,
                word_repres, word_dim, sentence_repres, sentence_dim, seq_mask,
                edgelabel_vocab, is_training=is_training)

        # collect placeholders
        self.sentence_words = self.encoder.sentence_words
        self.sentence_lengths = self.encoder.sentence_lengths
        if options.with_char:
            self.sentence_chars = self.encoder.sentence_chars
            self.sentence_chars_lengths = self.encoder.sentence_chars_lengths
        if options.with_POS:
            self.sentence_POSs = self.encoder.sentence_POSs

        self.in_neigh_indices = self.graph_encoder.in_neighbor_indices
        self.in_neigh_edges = self.graph_encoder.in_neighbor_edges
        self.in_neigh_mask = self.graph_encoder.in_neighbor_mask

        self.out_neigh_indices = self.graph_encoder.out_neighbor_indices
        self.out_neigh_edges = self.graph_encoder.out_neighbor_edges
        self.out_neigh_mask = self.graph_encoder.out_neighbor_mask

        if options.forest_prob_aware and options.forest_type != '1best':
            self.in_neigh_prob = self.graph_encoder.in_neighbor_prob
            self.out_neigh_prob = self.graph_encoder.out_neighbor_prob

        self.entity_indices = tf.placeholder(tf.int32, [None, None, None],
                                name="entity_indices")
        self.entity_indices_mask = tf.placeholder(tf.float32, [None, None, None],
                                name="entity_indices_mask")

        # collect inputs for final classifier
        final_repres = self.graph_encoder.graph_hiddens
        final_shape = tf.shape(final_repres)
        batch_size = final_shape[0]
        sentence_size_max = final_shape[1]

        # [batch, 2, indices, sentence_dim]
        entity_repres = collect_by_indices(final_repres, self.entity_indices)
        entity_repres = entity_repres * tf.expand_dims(self.entity_indices_mask, axis=-1)
        # [batch, 2, sentence_dim]
        entity_repres = tf.reduce_mean(entity_repres, axis=2)
        # [batch, 2*sentence_dim]
        entity_repres = tf.reshape(entity_repres, [batch_size, 2*sentence_dim])

        ### regarding Zhang et al., EMNLP 2018
        #h_sent = tf.reduce_max(final_repres, axis=1)
        #hsent_loss = None
        #if options.lambda_l2_hsent > 0.0:
        #    hsent_loss = tf.reduce_mean(
        #            tf.reduce_sum(h_sent * h_sent, axis=-1), axis=-1)
        #h_s = tf.reduce_max(
        #        range_repres(final_repres, sentence_size_max, self.sbj_starts, self.sbj_ends),
        #        axis=1)
        #h_o = tf.reduce_max(
        #        range_repres(final_repres, sentence_size_max, self.obj_starts, self.obj_ends),
        #        axis=1)
        #h_final = tf.concat([h_sent, h_s, h_o], axis=1) # [batch, sentence_dim*3]
        #h_final = tf.layers.dense(h_final, options.ffnn_size, name="ffnn_1", activation=tf.nn.relu) # [batch, ffnn_size]
        #h_final = tf.layers.dense(h_final, options.ffnn_size, name="ffnn_2", activation=tf.nn.relu) # [batch, ffnn_size]

        ## [batch, class_num]
        self.distribution = _clip_and_normalize(tf.layers.dense(entity_repres, options.num_relations,
                name="ffnn_out", activation=tf.nn.softmax), 1.0e-6)
        self.rsts = tf.argmax(self.distribution, axis=-1, output_type=tf.int32)

        ## calculating accuracy
        self.refs = tf.placeholder(tf.int32, [None,])
        self.accu = tf.reduce_sum(
                tf.cast(tf.equal(self.rsts, self.refs), dtype=tf.float32))

        ## calculating loss
        # xent: [batch]
        xent = -tf.reduce_sum(
                tf.one_hot(self.refs, options.num_relations) * tf.log(self.distribution),
                axis=-1)

        self.loss = tf.reduce_mean(xent)

        if mode != 'train':
            print('Return from here, just evaluate')
            return

        ### NER loss

        self.ne_refs = tf.placeholder(tf.int32, [None,None,])

        ne_distribution = _clip_and_normalize(tf.layers.dense(final_repres, options.num_nes,
                name="ffnn_ne", activation=tf.nn.softmax), 1.0e-6)

        # [batch, sentence_num]
        ne_xent = -tf.reduce_sum(
                tf.one_hot(self.ne_refs, options.num_nes) * tf.log(ne_distribution),
                axis=-1)

        ne_loss = tf.reduce_mean(ne_xent)
        self.loss += ne_loss

        #if options.lambda_l2_hsent > 0.0:
        #    self.loss += hsent_loss * options.lambda_l2_hsent

        clipper = 5
        tvars = tf.trainable_variables()
        if options.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss += options.lambda_l2 * l2_loss

        if hasattr(options, "decay") and options.decay != "none":
            global_step = tf.Variable(0, trainable=False)
            if options.decay == 'piece':
                values, bounds = [options.learning_rate,], []
                for i in range(10):
                    values.append(values[-1]*0.9)
                    bounds.append(options.trn_bch_num*10*i)
                learning_rate = tf.train.piecewise_constant(global_step, bounds, values)
            elif options.decay == 'poly':
                decay_steps = options.trn_bch_num*options.max_epochs
                learning_rate = tf.train.polynomial_decay(options.learning_rate, global_step, decay_steps,
                        end_learning_rate=0.1*options.learning_rate, power=0.5)
            elif options.decay == 'cos':
                decay_steps = options.trn_bch_num*options.max_epochs
                learning_rate = tf.train.cosine_decay(options.learning_rate, global_step, decay_steps,
                        alpha = 0.1)
            else:
                assert False, 'not supported'
        else:
            global_step = None
            learning_rate = options.learning_rate

        if options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            assert False, 'not supported optimize type'
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        extra_train_ops = []
        train_ops = [train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    def execute(self, sess, batch, options, is_train=False):
        feed_dict = {}
        feed_dict[self.sentence_words] = batch.sentence_words
        feed_dict[self.sentence_lengths] = batch.sentence_lengths
        if options.with_char:
            feed_dict[self.sentence_chars] = batch.sentence_chars
            feed_dict[self.sentence_chars_lengths] = batch.sentence_chars_lengths
        if options.with_POS:
            feed_dict[self.sentence_POSs] = batch.sentence_POSs

        feed_dict[self.in_neigh_indices] = batch.in_neigh_indices
        feed_dict[self.in_neigh_edges] = batch.in_neigh_edges
        feed_dict[self.in_neigh_mask] = batch.in_neigh_mask

        feed_dict[self.out_neigh_indices] = batch.out_neigh_indices
        feed_dict[self.out_neigh_edges] = batch.out_neigh_edges
        feed_dict[self.out_neigh_mask] = batch.out_neigh_mask

        if options.forest_prob_aware and options.forest_type != '1best':
            feed_dict[self.in_neigh_prob] = batch.in_neigh_prob
            feed_dict[self.out_neigh_prob] = batch.out_neigh_prob

        feed_dict[self.entity_indices] = batch.entity_indices
        feed_dict[self.entity_indices_mask] = batch.entity_indices_mask
        feed_dict[self.refs] = batch.refs

        if is_train:
            feed_dict[self.ne_refs] = batch.nes
            return sess.run([self.rsts, self.accu, self.loss, self.train_op], feed_dict)
        else:
            return sess.run([self.rsts, self.accu, self.loss], feed_dict)


if __name__ == '__main__':
    pass
