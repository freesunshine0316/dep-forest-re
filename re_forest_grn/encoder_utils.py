import tensorflow as tf


def collect_final_step_lstm(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')


class SeqEncoder(object):
    def __init__(self, options, word_vocab=None, char_vocab=None, POS_vocab=None):

        self.options = options

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.POS_vocab = POS_vocab

        self.sentence_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, sentence_len]
        self.sentence_lengths = tf.placeholder(tf.int32, [None])

        if options.with_char:
            self.sentence_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, sentence_len, word_len]
            self.sentence_chars_lengths = tf.placeholder(tf.int32, [None, None]) # [batch_size, sentence_len]

        if options.with_POS:
            self.sentence_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, sentence_len]


    def encode(self, is_training=True, only_word_repre=False):
        options = self.options

        input_shape = tf.shape(self.sentence_words)
        batch_size = input_shape[0]
        sentence_len = input_shape[1]

        # ======word repres layer======
        word_repres = []
        word_dim = 0

        if options.fix_word_vec:
            word_vec_trainable = False
            cur_device = '/cpu:0'
        else:
            word_vec_trainable = True
            cur_device = '/gpu:0'
        with tf.variable_scope("embedding"), tf.device(cur_device):
            self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                    initializer=tf.constant(self.word_vocab.word_vecs), dtype=tf.float32)

        # [batch_size, sentence_len, word_dim]
        word_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.sentence_words)
        word_repres.append(word_word_repres)
        word_dim += self.word_vocab.word_dim

        if options.with_char:
            assert self.char_vocab is not None
            word_len = tf.shape(self.sentence_chars)[2]
            char_dim = self.char_vocab.word_dim
            self.char_embedding = tf.get_variable("char_embedding",
                    initializer=tf.constant(self.char_vocab.word_vecs), dtype=tf.float32)

            # [batch_size, sentence_len, word_len, char_dim]
            word_chars_repres = tf.nn.embedding_lookup(self.char_embedding,
                    self.sentence_chars)
            word_chars_repres = tf.reshape(word_chars_repres,
                    shape=[-1, word_len, char_dim])
            word_chars_lengths = tf.reshape(self.sentence_chars_lengths, [-1])
            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(options.char_lstm_dim)
                if is_training:
                    char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell,
                            output_keep_prob=(1 - options.dropout_rate))
                char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])

                # [batch_size*sentence_len, word_len, char_lstm_dim]
                word_chars_repres = tf.nn.dynamic_rnn(char_lstm_cell, word_chars_repres,
                        sequence_length=word_chars_lengths, dtype=tf.float32)[0]
                # [batch_size*sentence_len, char_lstm_dim]
                word_chars_repres = collect_final_step_lstm(word_chars_repres, word_chars_lengths-1)
                # [batch_size, sentence_len, char_lstm_dim]
                word_chars_repres = tf.reshape(word_chars_repres, [batch_size, sentence_len, options.char_lstm_dim])

            word_repres.append(word_chars_repres)
            word_dim += options.char_lstm_dim

        if options.with_POS:
            assert self.POS_vocab is not None
            self.POS_embedding = tf.get_variable("POS_embedding",
                    initializer=tf.constant(self.POS_vocab.word_vecs), dtype=tf.float32)
            word_POS_repres = tf.nn.embedding_lookup(self.POS_embedding,
                    self.sentence_POSs) # [batch_size, sentence_len, POS_dim]
            word_repres.append(word_POS_repres)
            word_dim += self.POS_vocab.word_dim

        word_repres = tf.concat(word_repres, 2) # [batch_size, sentence_len, dim]

        if options.compress_input: # compress input word vector into smaller vectors
            w_compress = tf.get_variable("w_compress", [word_dim, options.compress_input_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress", [options.compress_input_dim], dtype=tf.float32)

            word_repres = tf.reshape(word_repres, [-1, word_dim])
            word_repres = tf.matmul(word_repres, w_compress) + b_compress
            word_repres = tf.tanh(word_repres)
            word_repres = tf.reshape(word_repres, [batch_size, sentence_len, options.compress_input_dim])
            word_dim = options.compress_input_dim

        if is_training:
            word_repres = tf.nn.dropout(word_repres, (1 - options.dropout_rate))

        seq_mask = tf.sequence_mask(self.sentence_lengths, sentence_len, dtype=tf.float32) # [batch_size, sentence_len]

        word_repres = word_repres * tf.expand_dims(seq_mask, axis=-1)

        # sequential context matching
        sentence_fw = None
        sentence_bw = None
        context_repres = []
        context_dim = 0
        with tf.variable_scope('biLSTM'):
            cur_context_repres = word_repres
            for i in xrange(options.num_lstm_layer):
                with tf.variable_scope('layer-{}'.format(i)):
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.LSTMCell(options.hidden_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.LSTMCell(options.hidden_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw,
                                output_keep_prob=(1 - options.dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw,
                                output_keep_prob=(1 - options.dropout_rate))

                    # context repres
                    ((context_repres_fw, context_repres_bw), (lstm_fw, lstm_bw)) = tf.nn.bidirectional_dynamic_rnn(
                                context_lstm_cell_fw, context_lstm_cell_bw, cur_context_repres, dtype=tf.float32,
                                sequence_length=self.sentence_lengths) # [batch_size, sentence_len, lstm_dim]
                    # [batch_size, sentence_len, 2*lstm_dim]
                    cur_context_repres = tf.concat([context_repres_fw, context_repres_bw], 2)
                    context_dim += 2*options.hidden_dim
                    context_repres.append(cur_context_repres)

        # [batch_size, sentence_len, 2*L*lstm_dim]
        if context_dim > 0:
            context_repres = tf.concat(context_repres, 2)
        else:
            context_dim = 2*options.hidden_dim
            context_repres = tf.zeros([batch_size, sentence_len, context_dim])

        if is_training:
            context_repres = tf.nn.dropout(context_repres,
                    (1 - options.dropout_rate))
        context_repres = context_repres * tf.expand_dims(seq_mask, axis=-1)

        # (word_repres, word_dim, word_repres, word_dim, seq_mask)
        return (word_repres, word_dim, context_repres, context_dim, seq_mask)

