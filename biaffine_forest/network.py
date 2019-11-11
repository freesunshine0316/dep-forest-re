#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle as pkl

import numpy as np
import tensorflow as tf

from lib import models
from lib import optimizers
from lib import rnn_cells

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset

import json, codecs

#***************************************************************
class Network(Configurable):
  """"""

  #=============================================================
  def __init__(self, model, *args, **kwargs):
    """"""

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')

    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(Network, self).__init__(*args, **kwargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    with open(os.path.join(self.save_dir, 'config.cfg'), 'w') as f:
      self._config.write(f)

    self._global_step = tf.Variable(0., trainable=False)
    self._global_epoch = tf.Variable(0., trainable=False)
    self._model = model(self._config, global_step=self.global_step)

    self._vocabs = []
    vocab_files = [(self.word_file, 1, 'Words'),
                   (self.tag_file, [3, 4], 'Tags'),
                   (self.rel_file, 7, 'Rels')]
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    use_pretrained=(not i),
                    global_step=self.global_step)
      self._vocabs.append(vocab)
    if self.use_forest:
      self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset', forest_file=self.train_forest_file)
      self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset', forest_file=self.valid_forest_file)
      self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset', forest_file=self.test_forest_file)
    else:
      self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
      self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
      self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')

    self._ops = self._gen_ops()
    self._save_vars = filter(lambda x: u'Pretrained' not in x.name, tf.global_variables())
    self.history = {
      'train_loss': [],
      'train_accuracy': [],
      'valid_loss': [],
      'valid_accuracy': [],
      'test_acuracy': 0
    }
    return

  #=============================================================
  def train_minibatches(self):
    """"""

    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          self.model.forest_idxs)

  #=============================================================
  def valid_minibatches(self):
    """"""

    return self._validset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          self.model.forest_idxs,
                                          shuffle=False)

  #=============================================================
  def test_minibatches(self):
    """"""

    return self._testset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          self.model.forest_idxs,
                                          shuffle=False)

  #=============================================================
  # assumes the sess has already been initialized
  def train(self, sess):
    """"""

    save_path = os.path.join(self.save_dir, self.name.lower() + '-pretrained')
    saver = tf.train.Saver(self.save_vars, max_to_keep=1)

    n_bkts = self.n_bkts
    train_iters = self.train_iters
    print_every = self.print_every
    validate_every = self.validate_every
    save_every = self.save_every
    try:
      train_time = 0
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      total_train_iters = 0 #sess.run(self.global_step)
      valid_time = 0
      valid_loss = 0
      valid_accuracy = 0
      BEST_AVG_UAS_LAS = 0
      while total_train_iters < train_iters:
        for j, (feed_dict, _) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          start_time = time.time()
          _, loss, n_correct, n_tokens = sess.run(self.ops['train_op'], feed_dict=feed_dict)
          train_time += time.time() - start_time
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_tokens += n_tokens
          n_train_iters += 1
          total_train_iters += 1

          self.history['train_loss'].append(loss)
          self.history['train_accuracy'].append(100 * n_correct / n_tokens)
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            valid_time = 0
            valid_loss = 0
            n_valid_sents = 0
            n_valid_correct = 0
            n_valid_tokens = 0
            with open(os.path.join(self.save_dir, 'sanitycheck.txt'), 'w') as f:
              for k, (feed_dict, _) in enumerate(self.valid_minibatches()):
                inputs = feed_dict[self._validset.inputs]
                targets = feed_dict[self._validset.targets]
                start_time = time.time()
                loss, n_correct, n_tokens, predictions = sess.run(self.ops['valid_op'], feed_dict=feed_dict)
                valid_time += time.time() - start_time
                valid_loss += loss
                n_valid_sents += len(targets)
                n_valid_correct += n_correct
                n_valid_tokens += n_tokens
                self.model.sanity_check(inputs, targets, predictions, self._vocabs, f, feed_dict=feed_dict)
            valid_loss /= k+1
            valid_accuracy = 100 * n_valid_correct / n_valid_tokens
            valid_time = n_valid_sents / valid_time
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_accuracy'].append(valid_accuracy)
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_tokens
            train_time = n_train_sents / train_time
            print('%6d) Train loss: %.4f    Train acc: %5.2f%%    Train rate: %6.1f sents/sec\n\tValid loss: %.4f    Valid acc: %5.2f%%    Valid rate: %6.1f sents/sec' % (total_train_iters, train_loss, train_accuracy, train_time, valid_loss, valid_accuracy, valid_time))
            sys.stdout.flush()
            train_time = 0
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_tokens = 0
            n_train_iters = 0
        sess.run(self._global_epoch.assign_add(1.))
        if save_every and (total_train_iters % save_every == 0):
          UAS, LAS = self.test(sess, validate=True)
          AVG_UAS_LAS = (UAS+LAS)/2
          if AVG_UAS_LAS > BEST_AVG_UAS_LAS:
            print("\t Exceed: ", AVG_UAS_LAS, BEST_AVG_UAS_LAS, UAS, LAS)
            BEST_AVG_UAS_LAS = AVG_UAS_LAS
            saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                           latest_filename=self.name.lower(),
                           global_step=self.global_epoch,
                           write_meta_graph=False)
            with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
                  pkl.dump(self.history, f)
            with open(os.path.join(self.save_dir, 'scores.txt'), 'w') as f:
                  pass
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)

    UAS, LAS = self.test(sess, validate=True)
    AVG_UAS_LAS = (UAS + LAS) / 2
    if AVG_UAS_LAS > BEST_AVG_UAS_LAS:
        print("\t Exceed: ", AVG_UAS_LAS, BEST_AVG_UAS_LAS, UAS, LAS)
        BEST_AVG_UAS_LAS = AVG_UAS_LAS
        saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                   latest_filename=self.name.lower(),
                   global_step=self.global_epoch,
                   write_meta_graph=False)
        with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
          pkl.dump(self.history, f)
        with open(os.path.join(self.save_dir, 'scores.txt'), 'w') as f:
          pass
    return

  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test_cube(self, sess, is_sparse):
    suffix_str = 'cubesparse' if is_sparse else 'cube'
    filename = self.test_file + '_%s.json' % suffix_str
    minibatches = self.test_minibatches
    dataset = self._testset
    op = self.ops['test_op'][1]

    # [buckets, sents, words, words, rels] for cube
    # [buckets, sents, triples, prb-hi-mi-lb] for cubesparse
    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    btch_idx = 0
    for (feed_dict, sents) in minibatches():
      print('Cube batch {}'.format(btch_idx))
      btch_idx += 1
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      mb_probs = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(self.model.validate_cube(mb_inputs, mb_targets, mb_probs, is_sparse, self.rels))
      all_sents[-1].extend([[w.decode('utf-8') for w in sent] for sent in sents])
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    all_new_predictions = []
    all_new_sents = []
    for bkt_idx, idx in dataset._metabucket.data:
      all_new_predictions.append(all_predictions[bkt_idx][idx])
      all_new_sents.append(all_sents[bkt_idx][idx])
    all_predictions = all_new_predictions
    all_sents = all_new_sents
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      json.dump((all_predictions, all_sents, suffix_str), f)
    return

  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test_nbest(self, sess):
    filename = self.test_file + '_nbest.json'
    minibatches = self.test_minibatches
    dataset = self._testset
    op = self.ops['test_op'][1]

    all_predictions = [[]] # [buckets, sents, nbests]
    all_sents = [[]]
    bkt_idx = 0
    btch_idx = 0
    for (feed_dict, sents) in minibatches():
      print('N-best batch {}'.format(btch_idx))
      btch_idx += 1
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      mb_probs = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(self.model.validate_nbest(mb_inputs, mb_targets, mb_probs, self.rels)) # [sents, nbests]
      all_sents[-1].extend([[w.decode('utf-8') for w in sent] for sent in sents])
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    all_new_predictions = []
    all_new_sents = []
    for bkt_idx, idx in dataset._metabucket.data:
      all_new_predictions.append(all_predictions[bkt_idx][idx])
      all_new_sents.append(all_sents[bkt_idx][idx])
    all_predictions = all_new_predictions
    all_sents = all_new_sents
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      json.dump((all_predictions, all_sents, 'nbest'), f)
    return

  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test(self, sess, validate=False):
    """"""

    if validate:
      filename = self.valid_file + '_1best.txt'
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.ops['test_op'][0]
    else:
      filename = self.test_file + '_1best.txt'
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.ops['test_op'][1]

    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    for (feed_dict, sents) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      mb_probs = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(self.model.validate(mb_inputs, mb_targets, mb_probs))
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      for bkt_idx, idx in dataset._metabucket.data:
        data = dataset._metabucket[bkt_idx].data[idx][1:]
        preds = all_predictions[bkt_idx][idx]
        words = all_sents[bkt_idx][idx]
        for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
          tup = (
            i+1,
            word,
            self.tags[pred[3]] if pred[3] != -1 else self.tags[datum[2]],
            self.tags[pred[4]] if pred[4] != -1 else self.tags[datum[3]],
            str(pred[5]) if pred[5] != -1 else str(datum[4]),
            self.rels[pred[6]] if pred[6] != -1 else self.rels[datum[5]],
            str(pred[7]) if pred[7] != -1 else '_',
            self.rels[pred[8]] if pred[8] != -1 else '_',
          )
          f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
        f.write('\n')
    with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
      s, correct = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)), punct=self.model.PUNCT)
      f.write(s)
      UAS, LAS = np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100
    return UAS, LAS

  #=============================================================
  def savefigs(self, sess, optimizer=False):
    """"""

    import gc
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    matdir = os.path.join(self.save_dir, 'matrices')
    if not os.path.isdir(matdir):
      os.mkdir(matdir)
    for var in self.save_vars:
      if optimizer or ('Optimizer' not in var.name):
        print(var.name)
        mat = sess.run(var)
        if len(mat.shape) == 1:
          mat = mat[None,:]
        plt.figure()
        try:
          plt.pcolor(mat, cmap='RdBu')
          plt.gca().invert_yaxis()
          plt.colorbar()
          plt.clim(vmin=-1, vmax=1)
          plt.title(var.name)
          plt.savefig(os.path.join(matdir, var.name.replace('/', '-')))
        except ValueError:
          pass
        plt.close()
        del mat
        gc.collect()

  #=============================================================
  def _gen_ops(self):
    """"""

    optimizer = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    train_output = self._model(self._trainset)

    train_op = optimizer.minimize(train_output['loss'])
    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=optimizer)
    test_output = self._model(self._testset, moving_params=optimizer)

    ops = {}
    ops['train_op'] = [train_op,
                       train_output['loss'],
                       train_output['n_correct'],
                       train_output['n_tokens']
                       ]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['n_correct'],
                       valid_output['n_tokens'],
                       valid_output['predictions']]
    ops['test_op'] = [valid_output['probabilities'],
                      test_output['probabilities']]
    ops['optimizer'] = optimizer

    return ops

  #=============================================================
  @property
  def global_step(self):
    return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def model(self):
    return self._model
  @property
  def words(self):
    return self._vocabs[0]
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def ops(self):
    return self._ops
  @property
  def save_vars(self):
    return self._save_vars

#***************************************************************
if __name__ == '__main__':
  """"""

  import argparse

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--load', action='store_true')
  argparser.add_argument('--model', default='GRNParser')
  argparser.add_argument('--matrix', action='store_true')
  argparser.add_argument('--nbest', action='store_true')
  argparser.add_argument('--cube', action='store_true')
  argparser.add_argument('--cubesparse', action='store_true')

  args, extra_args = argparser.parse_known_args()
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}

  print('*** '+args.model+' ***')
  model = getattr(models, args.model)

  print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

  test_cases = (args.test or args.matrix or args.nbest or args.cube or args.cubesparse)
  other_cases = test_cases or args.load
  if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not other_cases:
    raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  if other_cases and 'save_dir' in cargs:
    cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')
  network = Network(model, **cargs)
  os.system('echo Model: %s > %s/MODEL' % (network.model.__class__.__name__, network.save_dir))
  #print([v.name for v in network.save_vars])
  config_proto = tf.ConfigProto()
  #config_proto.gpu_options.per_process_gpu_memory_fraction = network.per_process_gpu_memory_fraction
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    if not test_cases: # training
      if args.load: # continue from previous save point
        os.system('echo Training: > %s/HEAD' % network.save_dir)
        os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
        saver = tf.train.Saver(var_list=network.save_vars)
        saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
        if os.path.isfile(os.path.join(network.save_dir, 'history.pkl')):
          with open(os.path.join(network.save_dir, 'history.pkl')) as f:
            network.history = pkl.load(f)
      else:
        os.system('echo Loading: >> %s/HEAD' % network.save_dir)
        os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
      network.train(sess)
    elif args.matrix:
      saver = tf.train.Saver(var_list=network.save_vars)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      network.savefigs(sess)
    elif args.nbest:
      saver = tf.train.Saver(var_list=network.save_vars)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      start_time = time.time()
      network.test_nbest(sess)
      print('Parsing took %f seconds' % (time.time() - start_time))
    elif args.cube or args.cubesparse:
      is_sparse = args.cubesparse
      saver = tf.train.Saver(var_list=network.save_vars)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      start_time = time.time()
      network.test_cube(sess, is_sparse)
      print('Parsing took %f seconds' % (time.time() - start_time))
    else:
      os.system('echo Testing: >> %s/HEAD' % network.save_dir)
      os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
      saver = tf.train.Saver(var_list=network.save_vars)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      network.test(sess, validate=True)
      start_time = time.time()
      network.test(sess, validate=False)
      print('Parsing took %f seconds' % (time.time() - start_time))
