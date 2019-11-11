# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import codecs
import json
import random

from vocab_utils import Vocab
import namespace_utils
import G2S_data_stream
from G2S_model_graph import ModelGraph

FLAGS = None
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


import platform
def get_machine_name():
    return platform.node()

def vec2string(val):
    result = ""
    for v in val:
        result += " {}".format(v)
    return result.strip()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cal_f1(both, ref, out):
    p = both/out if out > 0.0 else 0.0
    r = both/ref if ref > 0.0 else 0.0
    r = r*0.9225
    f = 2*p*r/(p+r) if min(p,r) > 0.0 else 0.0
    return p, r, f


def evaluate(sess, valid_graph, devDataStream, options):
    norel_id = 0
    devDataStream.reset()
    dev_loss = 0.0
    dev_both = 0.0
    dev_ref = 0.0
    dev_out = 0.0
    for batch_index in xrange(devDataStream.get_num_batch()): # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        output_value, _, loss_value  = valid_graph.execute(sess, cur_batch, options, is_train=False)
        output_value = output_value.flatten().tolist()
        dev_loss += loss_value
        for i in range(cur_batch.batch_size):
            if cur_batch.refs[i] != norel_id:
                dev_ref += 1.0
            if output_value[i] != norel_id:
                dev_out += 1.0
                if output_value[i] == cur_batch.refs[i]:
                    dev_both += 1.0

    print('###### both {}, ref {}, out {}'.format(dev_both, dev_ref, dev_out))
    dev_precision, dev_recall, dev_f1 = cal_f1(dev_both, dev_ref, dev_out)
    return {'dev_loss':dev_loss, 'dev_f1':dev_f1, 'dev_precision':dev_precision, 'dev_recall':dev_recall}


def main(_):
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/G2S.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    print('Loading data.')
    trainset = G2S_data_stream.read_bionlp_file(FLAGS.train_path, FLAGS.train_dep_path, FLAGS)
    devset = G2S_data_stream.read_bionlp_file(FLAGS.dev_path, FLAGS.dev_dep_path, FLAGS)
    if FLAGS.dev_from == 'shuffle':
        random.shuffle(devset)
    elif FLAGS.dev_from == 'last':
        devset.reverse()
    N = int(len(devset)*FLAGS.dev_percent)
    trainset += devset[N:]
    devset = devset[:N]
    print('Number of training samples: {}'.format(len(trainset)))
    print('Number of dev samples: {}'.format(len(devset)))
    print('Number of relations: {}'.format(FLAGS.num_relations))
    print('Number of NEs: {}'.format(FLAGS.num_nes))

    word_vocab = None
    char_vocab = None
    POS_vocab = None
    edgelabel_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + ".best.model"
    if os.path.exists(best_path + ".index"):
        has_pretrained_model = True
        print('!!Existing pretrained model. Loading vocabs.')
        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        if FLAGS.with_char:
            char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
            print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
        if FLAGS.with_POS:
            POS_vocab = Vocab(path_prefix + ".POS_vocab", fileformat='txt2')
            print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
        edgelabel_vocab = Vocab(path_prefix + ".edgelabel_vocab", fileformat='txt2')
        print('edgelabel_vocab: {}'.format(edgelabel_vocab.word_vecs.shape))
    else:
        print('Collecting vocabs.')
        all_words = set()
        all_chars = set()
        all_poses = set()
        all_edgelabels = set()
        G2S_data_stream.collect_vocabs(trainset, all_words, all_chars, all_poses, all_edgelabels)
        G2S_data_stream.collect_vocabs(devset, all_words, all_chars, all_poses, all_edgelabels)
        print('Number of words: {}'.format(len(all_words)))
        print('Number of chars: {}'.format(len(all_chars)))
        print('Number of poses: {}'.format(len(all_poses)))
        print('Number of edgelabels: {}'.format(len(all_edgelabels)))

        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        if FLAGS.with_char:
            char_vocab = Vocab(voc=all_chars, dim=FLAGS.char_dim, fileformat='build')
            char_vocab.dump_to_txt2(path_prefix + ".char_vocab")
        if FLAGS.with_POS:
            POS_vocab = Vocab(voc=all_poses, dim=FLAGS.POS_dim, fileformat='build')
            POS_vocab.dump_to_txt2(path_prefix + ".POS_vocab")
        edgelabel_vocab = Vocab(voc=all_edgelabels, dim=FLAGS.edgelabel_dim, fileformat='build')
        edgelabel_vocab.dump_to_txt2(path_prefix + ".edgelabel_vocab")

    print('word vocab size {}'.format(word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build DataStream ... ')
    trainDataStream = G2S_data_stream.G2SDataStream(FLAGS, trainset, word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
            isShuffle=True, isLoop=True, isSort=True, is_training=True)

    devDataStream = G2S_data_stream.G2SDataStream(FLAGS, devset, word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
            isShuffle=False, isLoop=False, isSort=True)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    FLAGS.trn_bch_num = trainDataStream.get_num_batch()

    # initialize the best bleu and accu scores for current training session
    best_accu = FLAGS.best_accu if FLAGS.__dict__.has_key('best_accu') else 0.0
    if best_accu > 0.0:
        print('With initial dev accuracy {}'.format(best_accu))

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
                                         FLAGS, mode='train')

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(word_vocab, char_vocab, POS_vocab, edgelabel_vocab,
                                         FLAGS, mode='evaluate')

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        if has_pretrained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

            if abs(best_accu) < 1e-5:
                print("Getting ACCU score for the model")
                best_accu = evaluate(sess, valid_graph, devDataStream, FLAGS)['dev_f1']
                FLAGS.best_accu = best_accu
                namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                print('ACCU = %.4f' % best_accu)
                log_file.write('ACCU = %.4f\n' % best_accu)

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        last_step = 0
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            cur_batch = trainDataStream.nextBatch()
            _, _, cur_loss, _ = train_graph.execute(sess, cur_batch, FLAGS, is_train=True)
            total_loss += cur_loss

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss/(step-last_step), duration))
                log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss/(step-last_step), duration))
                sys.stdout.flush()
                log_file.flush()
                last_step = step
                total_loss = 0.0

                # Evaluate against the validation set.
                start_time = time.time()
                print('Validation Data Eval:')
                res_dict = evaluate(sess, valid_graph, devDataStream, FLAGS)
                dev_loss = res_dict['dev_loss']
                dev_accu = res_dict['dev_f1']
                dev_precision = res_dict['dev_precision']
                dev_recall = res_dict['dev_recall']
                print('Dev loss = %.4f' % dev_loss)
                log_file.write('Dev loss = %.4f\n' % dev_loss)
                print('Dev F1 = %.4f, P = %.4f, R = %.4f' % (dev_accu, dev_precision, dev_recall))
                log_file.write('Dev F1 = %.4f, P =  %.4f, R = %.4f\n' % (dev_accu, dev_precision, dev_recall))
                log_file.flush()
                if best_accu < dev_accu:
                    print('Saving weights, ACCU {} (prev_best) < {} (cur)'.format(best_accu, dev_accu))
                    saver.save(sess, best_path)
                    best_accu = dev_accu
                    FLAGS.best_accu = dev_accu
                    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                duration = time.time() - start_time
                print('Duration %.3f sec' % (duration))
                sys.stdout.flush()

                log_file.write('Duration %.3f sec\n' % (duration))
                log_file.flush()
                start_time = time.time()

    log_file.close()

def enrich_options(options):

    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="2"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    FLAGS = enrich_options(FLAGS)

    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
