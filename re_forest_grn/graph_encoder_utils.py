import tensorflow as tf


def collect_neighbor(representation, positions):
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


def collect_neighbor_v2(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    batch_size = tf.shape(positions)[0]
    node_num = tf.shape(positions)[1]
    neigh_num = tf.shape(positions)[2]
    rids = tf.range(0, limit=batch_size) # [batch]
    rids = tf.reshape(rids, [-1, 1, 1]) # [batch, 1, 1]
    rids = tf.tile(rids, [1, node_num, neigh_num]) # [batch, nodes, neighbors]
    indices = tf.stack((rids, positions), axis=3) # [batch, nodes, neighbors, 2]
    return tf.gather_nd(representation, indices) # [batch, nodes, neighbors, feature_dim]


class GraphEncoder(object):
    def __init__(self, options, word_repres, word_dim, sentence_repres, sentence_dim, seq_mask,
            edgelabel_vocab, is_training=True):

        # [batch_size, nodes_size_max, neighbors_size_max]
        self.in_neighbor_indices = tf.placeholder(tf.int32, [None, None, None])
        self.in_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.in_neighbor_mask = tf.placeholder(tf.float32, [None, None, None])

        # [batch_size, nodes_size_max, neighbors_size_max]
        self.out_neighbor_indices = tf.placeholder(tf.int32, [None, None, None])
        self.out_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.out_neighbor_mask = tf.placeholder(tf.float32, [None, None, None])

        if options.forest_prob_aware and options.forest_type != '1best':
            self.in_neighbor_prob = tf.placeholder(tf.float32, [None, None, None])
            self.out_neighbor_prob = tf.placeholder(tf.float32, [None, None, None])

        # shapes
        input_shape = tf.shape(self.in_neighbor_indices)
        batch_size = input_shape[0]
        sentence_size_max = input_shape[1]
        in_neighbor_size_max = input_shape[2]
        out_neighbor_size_max = tf.shape(self.out_neighbor_indices)[2]

        # embeddings
        self.edge_embedding = tf.get_variable("edge_embedding",
                initializer=tf.constant(edgelabel_vocab.word_vecs), dtype=tf.float32)
        edge_dim = edgelabel_vocab.word_dim

        if options.with_edgelabel:
            u_input_dim = sentence_dim + edge_dim
        else:
            u_input_dim = sentence_dim


        with tf.variable_scope('grn_encoder'):
            # ==== input from in neighbors
            # [batch_size, sentence_len, neighbors_size_max, edge_dim]
            in_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
                    self.in_neighbor_edges)
            # [batch_size, sentence_len, neighbors_size_max, word_dim]
            in_neighbor_word_representations = collect_neighbor_v2(word_repres,
                    self.in_neighbor_indices)
            # [batch_size, sentence_len, neighbors_size_max, word_dim + edge_dim]
            in_neighbor_representations = tf.concat(
                    [in_neighbor_word_representations, in_neighbor_edge_representations], 3)
            if options.forest_prob_aware and options.forest_type != '1best':
                in_neighbor_representations = tf.multiply(in_neighbor_representations,
                        tf.expand_dims(self.in_neighbor_prob, axis=-1))
            in_neighbor_representations = tf.multiply(in_neighbor_representations,
                    tf.expand_dims(self.in_neighbor_mask, axis=-1))
            # [batch_size, sentence_len, word_dim + edge_dim]
            in_neighbor_representations = tf.reduce_sum(in_neighbor_representations, axis=2)
            in_neighbor_representations = tf.reshape(in_neighbor_representations,
                    [-1, word_dim + edge_dim])


            # ==== input from out neighbors
            # [batch_size, sentence_len, neighbors_size_max, edge_dim]
            out_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
                    self.out_neighbor_edges)
            # [batch_size, sentence_len, neighbors_size_max, word_dim]
            out_neighbor_word_representations = collect_neighbor_v2(word_repres,
                    self.out_neighbor_indices)
            # [batch_size, sentence_len, neighbors_size_max, word_dim + edge_dim]
            out_neighbor_representations = tf.concat(
                    [out_neighbor_word_representations, out_neighbor_edge_representations], 3)
            if options.forest_prob_aware and options.forest_type != '1best':
                out_neighbor_representations = tf.multiply(out_neighbor_representations,
                        tf.expand_dims(self.out_neighbor_prob, axis=-1))
            out_neighbor_representations = tf.multiply(out_neighbor_representations,
                    tf.expand_dims(self.out_neighbor_mask, axis=-1))
            # [batch_size, sentence_len, word_dim + edge_dim]
            out_neighbor_representations = tf.reduce_sum(out_neighbor_representations, axis=2)
            out_neighbor_representations = tf.reshape(out_neighbor_representations,
                    [-1, word_dim + edge_dim])


            node_hidden = sentence_repres
            node_cell = tf.zeros([batch_size, sentence_size_max, sentence_dim])

            w_in_ingate = tf.get_variable("w_in_ingate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_in_ingate = tf.get_variable("u_in_ingate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)
            b_ingate = tf.get_variable("b_in_ingate",
                    [sentence_dim], dtype=tf.float32)
            w_out_ingate = tf.get_variable("w_out_ingate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_out_ingate = tf.get_variable("u_out_ingate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)

            w_in_forgetgate = tf.get_variable("w_in_forgetgate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_in_forgetgate = tf.get_variable("u_in_forgetgate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)
            b_forgetgate = tf.get_variable("b_in_forgetgate",
                    [sentence_dim], dtype=tf.float32)
            w_out_forgetgate = tf.get_variable("w_out_forgetgate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_out_forgetgate = tf.get_variable("u_out_forgetgate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)

            w_in_outgate = tf.get_variable("w_in_outgate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_in_outgate = tf.get_variable("u_in_outgate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)
            b_outgate = tf.get_variable("b_in_outgate",
                    [sentence_dim], dtype=tf.float32)
            w_out_outgate = tf.get_variable("w_out_outgate",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_out_outgate = tf.get_variable("u_out_outgate",
                    [u_input_dim, sentence_dim], dtype=tf.float32)

            w_in_cell = tf.get_variable("w_in_cell",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_in_cell = tf.get_variable("u_in_cell",
                    [u_input_dim, sentence_dim], dtype=tf.float32)
            b_cell = tf.get_variable("b_in_cell",
                    [sentence_dim], dtype=tf.float32)
            w_out_cell = tf.get_variable("w_out_cell",
                    [word_dim + edge_dim, sentence_dim], dtype=tf.float32)
            u_out_cell = tf.get_variable("u_out_cell",
                    [u_input_dim, sentence_dim], dtype=tf.float32)

            word_repres = tf.reshape(word_repres, [-1, word_dim])

            # calculate question graph representation
            graph_representations = []
            for i in range(options.num_graph_layer):
                # =============== in neighbor hidden
                # [batch_size, sentence_len, neighbors_size_max, u_input_dim]
                in_neighbor_prev_hidden = collect_neighbor_v2(node_hidden,
                        self.in_neighbor_indices)
                if options.with_edgelabel:
                    in_neighbor_prev_hidden = tf.concat(
                            [in_neighbor_prev_hidden, in_neighbor_edge_representations], 3)
                in_neighbor_prev_hidden = tf.multiply(in_neighbor_prev_hidden,
                    tf.expand_dims(self.in_neighbor_mask, axis=-1))
                # [batch_size, sentence_len, u_input_dim]
                in_neighbor_prev_hidden = tf.reduce_sum(in_neighbor_prev_hidden, axis=2)
                in_neighbor_prev_hidden = tf.multiply(in_neighbor_prev_hidden,
                        tf.expand_dims(seq_mask, axis=-1))
                in_neighbor_prev_hidden = tf.reshape(in_neighbor_prev_hidden, [-1, u_input_dim])

                # =============== out neighbor hidden
                # [batch_size, sentence_len, neighbors_size_max, u_input_dim]
                out_neighbor_prev_hidden = collect_neighbor_v2(node_hidden,
                        self.out_neighbor_indices)
                if options.with_edgelabel:
                    out_neighbor_prev_hidden = tf.concat(
                            [out_neighbor_prev_hidden, out_neighbor_edge_representations], 3)
                out_neighbor_prev_hidden = tf.multiply(out_neighbor_prev_hidden,
                    tf.expand_dims(self.out_neighbor_mask, axis=-1))
                # [batch_size, sentence_len, u_input_dim]
                out_neighbor_prev_hidden = tf.reduce_sum(out_neighbor_prev_hidden, axis=2)
                out_neighbor_prev_hidden = tf.multiply(out_neighbor_prev_hidden,
                        tf.expand_dims(seq_mask, axis=-1))
                out_neighbor_prev_hidden = tf.reshape(out_neighbor_prev_hidden, [-1, u_input_dim])

                ## ig
                edge_ingate = tf.sigmoid(tf.matmul(in_neighbor_representations, w_in_ingate)
                                          + tf.matmul(in_neighbor_prev_hidden, u_in_ingate)
                                          + tf.matmul(out_neighbor_representations, w_out_ingate)
                                          + tf.matmul(out_neighbor_prev_hidden, u_out_ingate)
                                          + b_ingate)
                edge_ingate = tf.reshape(edge_ingate,
                        [batch_size, sentence_size_max, sentence_dim])
                ## fg
                edge_forgetgate = tf.sigmoid(tf.matmul(in_neighbor_representations, w_in_forgetgate)
                                          + tf.matmul(in_neighbor_prev_hidden, u_in_forgetgate)
                                          + tf.matmul(out_neighbor_representations, w_out_forgetgate)
                                          + tf.matmul(out_neighbor_prev_hidden, u_out_forgetgate)
                                          + b_forgetgate)
                edge_forgetgate = tf.reshape(edge_forgetgate,
                        [batch_size, sentence_size_max, sentence_dim])
                ## og
                edge_outgate = tf.sigmoid(tf.matmul(in_neighbor_representations, w_in_outgate)
                                          + tf.matmul(in_neighbor_prev_hidden, u_in_outgate)
                                          + tf.matmul(out_neighbor_representations, w_out_outgate)
                                          + tf.matmul(out_neighbor_prev_hidden, u_out_outgate)
                                          + b_outgate)
                edge_outgate = tf.reshape(edge_outgate,
                        [batch_size, sentence_size_max, sentence_dim])
                ## input
                edge_cell_input = tf.tanh(tf.matmul(in_neighbor_representations, w_in_cell)
                                          + tf.matmul(in_neighbor_prev_hidden, u_in_cell)
                                          + tf.matmul(out_neighbor_representations, w_out_cell)
                                          + tf.matmul(out_neighbor_prev_hidden, u_out_cell)
                                          + b_cell)
                edge_cell_input = tf.reshape(edge_cell_input,
                        [batch_size, sentence_size_max, sentence_dim])

                temp_cell = edge_forgetgate * node_cell + edge_ingate * edge_cell_input
                temp_hidden = edge_outgate * tf.tanh(temp_cell)
                #if is_training and i < options.num_graph_layer:
                #    temp_hidden = tf.nn.dropout(temp_hidden, (1 - options.dropout_rate))
                # apply mask
                node_cell = tf.multiply(temp_cell, tf.expand_dims(seq_mask, axis=-1))
                node_hidden = tf.multiply(temp_hidden, tf.expand_dims(seq_mask, axis=-1))

                graph_representations.append(node_hidden)

            # decide how to use graph_representations
            self.graph_representations = graph_representations
            self.graph_hiddens = node_hidden
            self.graph_cells = node_cell

