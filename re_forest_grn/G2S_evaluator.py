# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import re
import os
import sys
import json
import codecs
import time
import numpy as np
import collections
import tensorflow as tf
import namespace_utils

import G2S_trainer
import G2S_data_stream
from G2S_model_graph import ModelGraph

from vocab_utils import Vocab


tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--in_dep_path', type=str, required=True, help='The path to the test dependency file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the test file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    in_dep_path = args.in_dep_path
    out_path = args.out_path

    # We lose the cross-sentence positive cases by only considering single-sentence situations.
    # For convenient evaluation, we simply deduct those from the recall calculation.
    oracle_recall = 0.9225 if in_path.find('dev.json') >= 0 else 0.9254

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = G2S_trainer.enrich_options(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    char_vocab = None
    POS_vocab = None
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    if FLAGS.with_POS:
        POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
        print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
    edgelabel_vocab = Vocab(model_prefix + ".edgelabel_vocab", fileformat='txt2')
    print('edgelabel_vocab: {}'.format(edgelabel_vocab.word_vecs.shape))

    print('Loading test set from {}.'.format(in_path))
    if hasattr(FLAGS, 'num_relations') == False:
        FLAGS.num_relations = 2
    testset = G2S_data_stream.read_bionlp_file(in_path, in_dep_path, FLAGS)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    testDataStream = G2S_data_stream.G2SDataStream(FLAGS, testset, word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
            isShuffle=False, isLoop=False, isSort=False)
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

    init_scale = 0.01
    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
                                         FLAGS, mode="evaluate")

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        outputs = collections.defaultdict(list)
        norel_id = 0
        testDataStream.reset()
        start_time = time.time()
        both_num, out_num, ref_num = 0.0, 0.0, 0.0
        for batch_index in xrange(testDataStream.get_num_batch()): # for each batch
            cur_batch = testDataStream.get_batch(batch_index)
            output_value, _, _ = valid_graph.execute(sess, cur_batch, FLAGS, is_train=False)
            output_value = output_value.flatten().tolist()
            for i in range(cur_batch.batch_size):
                if cur_batch.instances[i][-2] != norel_id:
                    ref_num += 1.0
                if output_value[i] != norel_id:
                    out_num += 1.0
                    if output_value[i] == cur_batch.instances[i][-2]:
                        both_num += 1.0
                    inst = cur_batch.instances[i]
                    file_id, id1, id2 = inst[-1].split()
                    outputs[file_id].append([id1,id2,output_value[i]])
        duration = time.time() - start_time
        print('Decoding time %.3f sec' % (duration))
        precision = both_num / out_num
        recall = oracle_recall * both_num / ref_num
        fscore = 2*precision*recall/(precision+recall)
        print('P: {}, R: {}, F: {}'.format(precision, recall, fscore))
        json.dump(outputs, codecs.open(out_path, 'w', 'utf-8'))

