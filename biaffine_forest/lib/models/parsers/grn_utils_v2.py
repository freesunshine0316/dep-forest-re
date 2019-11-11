
import tensorflow as tf

def collect_neighbor_node_representations(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    feature_dim = tf.shape(representation)[2]
    input_shape = tf.shape(positions)
    batch_size = input_shape[0]
    num_nodes = input_shape[1]
    num_neighbors = input_shape[2]
    positions_flat = tf.reshape(positions, [batch_size, num_nodes*num_neighbors])
    def singel_instance(x):
        # x[0]: [num_nodes, feature_dim]
        # x[1]: [num_nodes*num_neighbors]
        return tf.gather(x[0], x[1])
    elems = (representation, positions_flat)
    representations = tf.map_fn(singel_instance, elems, dtype=tf.float32)
    return tf.reshape(representations, [batch_size, num_nodes, num_neighbors, feature_dim])

def collect_final_step_lstm(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')

def add_matrix_to_moving_average(matrix, moving_params=None):
    if moving_params is not None:
        matrix = moving_params.average(matrix)
    else:
        tf.add_to_collection('Weights', matrix)
    return matrix

def add_bias_to_moving_average(bias, moving_params=None):
    if moving_params is not None:
        bias = moving_params.average(bias)
    return bias

class GraphEncoder(object):
    # node_repre: previous RNN-layer output, current input [batch, sent, node_dim]
    # node_mask: [batch, sent, 1]
    # node_dim: vector size of previous RNN-layer output, scaler
    # in_neigh_idx: neighbor indices, [batch, sent, neigh_size]
    # in_neigh_rel_repre: neighbor (dependency) relation representation, [batch, sent, neigh_size, node_dim]
    # in_neigh_mask: [batch, sent, neigh_size]
    # out_neigh_idx: neighbor indices, [batch, sent, neigh_size]
    # out_neigh_rel_repre: neighbor (dependency) relation representation, [batch, sent, neigh_size, node_dim]
    # out_neigh_mask: [batch, sent, neigh_size]
    @classmethod
    def call(cls, node_repre, node_mask, in_neigh_idx, in_neigh_rel_repre, in_neigh_mask,
            out_neigh_idx, out_neigh_rel_repre, out_neigh_mask, keep_word_prob, num_syntax_match_layer, moving_params=None):
        # shapes
        input_shape = tf.shape(in_neigh_idx)
        batch_size = input_shape[0]
        node_size_max = input_shape[1]
        in_neigh_size_max = input_shape[2]
        out_neigh_size_max = tf.shape(out_neigh_idx)[2]
        node_dim = node_repre.get_shape().as_list()[-1]
        rel_dim = in_neigh_rel_repre.get_shape().as_list()[-1]
        reuse = (moving_params is not None)

        # apply the mask
        node_repre = node_repre * node_mask

        if keep_word_prob < 1.0 and not reuse:
            node_repre = tf.nn.dropout(node_repre, keep_word_prob)


        with tf.variable_scope('grn_encoder', reuse=reuse):
            node_hidden = node_repre
            node_cell = tf.zeros([batch_size, node_size_max, node_dim])

            w_trans = tf.get_variable("w_trans", [node_dim+rel_dim, node_dim], dtype=tf.float32)
            w_trans = add_matrix_to_moving_average(w_trans, moving_params)

            b_trans = tf.get_variable("b_trans", [node_dim], dtype=tf.float32)
            b_trans = add_bias_to_moving_average(b_trans, moving_params)

            ## parameters for LSTM gates
            u_in_ingate = tf.get_variable("u_in_ingate", [node_dim, node_dim], dtype=tf.float32)
            u_in_ingate = add_matrix_to_moving_average(u_in_ingate, moving_params)

            b_ingate = tf.get_variable("b_ingate", [node_dim], dtype=tf.float32)
            b_ingate = add_bias_to_moving_average(b_ingate, moving_params)

            u_out_ingate = tf.get_variable("u_out_ingate", [node_dim, node_dim], dtype=tf.float32)
            u_out_ingate = add_matrix_to_moving_average(u_out_ingate, moving_params)

            u_in_forgetgate = tf.get_variable("u_in_forgetgate", [node_dim, node_dim], dtype=tf.float32)
            u_in_forgetgate = add_matrix_to_moving_average(u_in_forgetgate, moving_params)

            b_forgetgate = tf.get_variable("b_forgetgate", [node_dim], dtype=tf.float32)
            b_forgetgate = add_bias_to_moving_average(b_forgetgate, moving_params)

            u_out_forgetgate = tf.get_variable("u_out_forgetgate", [node_dim, node_dim], dtype=tf.float32)
            u_out_forgetgate = add_matrix_to_moving_average(u_out_forgetgate, moving_params)

            u_in_outgate = tf.get_variable("u_in_outgate", [node_dim, node_dim], dtype=tf.float32)
            u_in_outgate =  add_matrix_to_moving_average(u_in_outgate, moving_params)

            b_outgate = tf.get_variable("b_outgate", [node_dim], dtype=tf.float32)
            b_outgate = add_bias_to_moving_average(b_outgate, moving_params)

            u_out_outgate = tf.get_variable("u_out_outgate", [node_dim, node_dim], dtype=tf.float32)
            u_out_outgate = add_matrix_to_moving_average(u_out_outgate, moving_params)

            u_in_cell = tf.get_variable("u_in_cell", [node_dim, node_dim], dtype=tf.float32)
            u_in_cell = add_matrix_to_moving_average(u_in_cell, moving_params)

            b_cell = tf.get_variable("b_cell", [node_dim], dtype=tf.float32)
            b_cell = add_bias_to_moving_average(b_cell, moving_params)

            u_out_cell = tf.get_variable("u_out_cell", [node_dim, node_dim], dtype=tf.float32)
            u_out_cell = add_matrix_to_moving_average(u_out_cell, moving_params)

            # calculate question graph representation
            graph_representations = []
            for i in xrange(num_syntax_match_layer):
                # =============== in edge hidden
                # [batch_size, node_len, neighbors_size, node_dim]
                in_neigh_prev_hidden = collect_neighbor_node_representations(node_hidden, in_neigh_idx)
                # [batch_size, node_len, neighbors_size, node_dim + rel_dim]
                in_neigh_prev_hidden = tf.concat(3, [in_neigh_prev_hidden, in_neigh_rel_repre])
                in_neigh_prev_hidden = tf.multiply(in_neigh_prev_hidden, tf.expand_dims(in_neigh_mask, axis=-1))
                # [batch_size, node_len, node_dim + rel_dim]
                in_neigh_prev_hidden = tf.reduce_sum(in_neigh_prev_hidden, axis=2)
                in_neigh_prev_hidden = tf.multiply(in_neigh_prev_hidden, node_mask)
                in_neigh_prev_hidden = tf.reshape(in_neigh_prev_hidden, [-1, node_dim+rel_dim])
                # [batch_size, node_len, node_dim]
                in_neigh_prev_hidden = tf.matmul(in_neigh_prev_hidden, w_trans) + b_trans
                in_neigh_prev_hidden = tf.tanh(in_neigh_prev_hidden)

                # =============== out edge hidden
                # h_{jk} [batch_size, node_len, neighbors_size, node_dim]
                out_neigh_prev_hidden = collect_neighbor_node_representations(node_hidden, out_neigh_idx)
                out_neigh_prev_hidden = tf.concat(3, [out_neigh_prev_hidden, out_neigh_rel_repre])
                out_neigh_prev_hidden = tf.multiply(out_neigh_prev_hidden, tf.expand_dims(out_neigh_mask, axis=-1))
                # [batch_size, node_len, node_dim]
                out_neigh_prev_hidden = tf.reduce_sum(out_neigh_prev_hidden, axis=2)
                out_neigh_prev_hidden = tf.multiply(out_neigh_prev_hidden, node_mask)
                out_neigh_prev_hidden = tf.reshape(out_neigh_prev_hidden, [-1, node_dim+rel_dim])
                # [batch_size, node_len, node_dim]
                out_neigh_prev_hidden = tf.matmul(out_neigh_prev_hidden, w_trans) + b_trans
                out_neigh_prev_hidden = tf.tanh(out_neigh_prev_hidden)

                ## ig
                ingate = tf.sigmoid(tf.matmul(in_neigh_prev_hidden, u_in_ingate)
                                          + tf.matmul(out_neigh_prev_hidden, u_out_ingate)
                                          + b_ingate)
                ingate = tf.reshape(ingate, [batch_size, node_size_max, node_dim])
                ## fg
                forgetgate = tf.sigmoid(tf.matmul(in_neigh_prev_hidden, u_in_forgetgate)
                                          + tf.matmul(out_neigh_prev_hidden, u_out_forgetgate)
                                          + b_forgetgate)
                forgetgate = tf.reshape(forgetgate, [batch_size, node_size_max, node_dim])
                ## og
                outgate = tf.sigmoid(tf.matmul(in_neigh_prev_hidden, u_in_outgate)
                                          + tf.matmul(out_neigh_prev_hidden, u_out_outgate)
                                          + b_outgate)
                outgate = tf.reshape(outgate, [batch_size, node_size_max, node_dim])
                ## input
                cell_input = tf.tanh(tf.matmul(in_neigh_prev_hidden, u_in_cell)
                                          + tf.matmul(out_neigh_prev_hidden, u_out_cell)
                                          + b_cell)
                cell_input = tf.reshape(cell_input, [batch_size, node_size_max, node_dim])

                new_node_cell = forgetgate * node_cell + ingate * cell_input
                new_node_hidden = outgate * tf.tanh(new_node_cell)
                # node mask
                # [batch_size, passage_len, node_dim]
                node_cell = tf.multiply(new_node_cell, node_mask)
                node_hidden = tf.multiply(new_node_hidden, node_mask)
                graph_representations.append(node_hidden)

            # decide how to use graph_representations
        return graph_representations, node_hidden, node_cell

