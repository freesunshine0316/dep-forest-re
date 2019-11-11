#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

import numpy as np
import tensorflow as tf
from collections import Counter

from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket
from forest_utils import load_cube, load_nbest, load_cubesparse
import sys
#***************************************************************
class Dataset(Configurable):
  """"""

  #=============================================================
  def __init__(self, filename, vocabs, builder, *args, **kwargs):
    """"""
    self.forest_file_name = kwargs.pop("forest_file", None)

    if self.forest_file_name is not None:
      print("[tlog] self.forest_file_name: " + self.forest_file_name)

    super(Dataset, self).__init__(*args, **kwargs)
    self.vocabs = vocabs
    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._forest_data = self.load_forest_file(self.forest_file_name)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.rebucket()

    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='targets')
    self.in_neighbor = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='in_neighbor') # [batch, word, neigh]
    self.in_neighbor_rel = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='in_neighbor_rel') # [batch, word, neigh]
    self.in_neighbor_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                      name='in_neighbor_mask')  # [batch, word, neigh]
    self.out_neighbor = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='out_neighbor')  # [batch, word, neigh]
    self.out_neighbor_rel = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                          name='out_neighbor_rel')  # [batch, word, neigh]
    self.out_neighbor_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                       name='out_neighbor_mask')  # [batch, word, neigh]
    self.builder = builder()

  #=============================================================
  def file_iterator(self, filename):
    """"""

    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line:
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if buff[-1]:
                  buff.append([])
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line:
            buff[-1].append(line)
          else:
            if buff[-1]:
              buff.append([])
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff

  #=============================================================
  def _remove_duplicate_items(self, node_index, neighbor, neighbor_rel, add_self=True, REL_UNK=2):
    unique_neighbor, unique_neighbor_rel = [], []
    node_cache = set()
    if add_self:
      unique_neighbor.append(node_index)
      unique_neighbor_rel.append(REL_UNK)
      node_cache.add((node_index, REL_UNK))
    for i in range(len(neighbor)):
      if (neighbor[i], neighbor_rel[i]) not in node_cache:
        unique_neighbor.append(neighbor[i])
        unique_neighbor_rel.append(neighbor_rel[i])
        node_cache.add((neighbor[i], neighbor_rel[i]))
    return unique_neighbor, unique_neighbor_rel

  # =============================================================
  def _process_buff(self, buff):
    """"""

    words, tags, rels = self.vocabs
    for i, sent in enumerate(buff):
      if self.use_forest:
        sent_str = tuple([token[1] for token in sent])
        triples, adj_lists = self._forest_data[len(sent_str)][sent_str]
        #print("[tlog] adj_lists: " + str(adj_lists))
        #sys.exit(0)
      for j, token in enumerate(sent):
        word, tag1, tag2, head, rel = token[words.conll_idx], \
                token[tags.conll_idx[0]], token[tags.conll_idx[1]], token[6], token[rels.conll_idx]
        if self.use_forest:
          #print("[tlog] adj_lists: " + str(adj_lists[0][j + 1]) + "\t" + str(adj_lists[1][j + 1]))
          unique_in_neighbor, unique_in_neighbor_rel = self._remove_duplicate_items(j+1,
                  adj_lists[0][j+1], adj_lists[1][j+1])
          unique_out_neighbor, unique_out_neighbor_rel = self._remove_duplicate_items(j+1,
                  adj_lists[2][j+1], adj_lists[3][j+1])
          #print("[tlog] adj_lists: " + str(adj_lists[0][j + 1]) + "\t" + str(adj_lists[1][j + 1]))
          #sys.exit(0)
          buff[i][j] = (word,) + words[word] + tags[tag1] + tags[tag2] + (int(head),) + rels[rel] + \
                  (unique_in_neighbor, unique_in_neighbor_rel, unique_out_neighbor, unique_out_neighbor_rel)
        else:
          buff[i][j] = (word,) + words[word] + tags[tag1] + tags[tag2] + (int(head),) + rels[rel]

      if self.use_forest:
        unique_in_neighbor, unique_in_neighbor_rel = self._remove_duplicate_items(0,
                adj_lists[0][0], adj_lists[1][0])
        unique_out_neighbor, unique_out_neighbor_rel = self._remove_duplicate_items(0,
                adj_lists[2][0], adj_lists[3][0])
        sent.insert(0, ('root', Vocab.ROOT, Vocab.ROOT, Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT, \
                        unique_in_neighbor, unique_in_neighbor_rel, unique_out_neighbor, unique_out_neighbor_rel))
      else:
        sent.insert(0, ('root', Vocab.ROOT, Vocab.ROOT, Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT))
    return buff

  # =============================================================
  def load_forest_file(self, forest_file_name):
    if forest_file_name is None or not self.use_forest:
      return
    if self.forest_type == 0:
      return load_nbest(forest_file_name, self.nbest_only_keep, self.vocabs[2])
    elif self.forest_type == 1:
      return load_cube(forest_file_name, self.cube_only_keep)
    elif self.forest_type == 2:
      return load_cube(forest_file_name, self.nbest_only_keep)
    elif self.forest_type == 3:
      return load_cubesparse(forest_file_name, self.cube_only_keep, self.vocabs[2])
    else:
      print("[Error] forest_type must be in [0, 1, 2]\n " +
            "\t 0 --- nbest, 10 \n" +
            "\t 1 --- cube, 0.05 \n" +
            "\t 2 --- cube, 10 \n" +
            "\t 3 --- cubesparse, 0.05 \n")
      sys.exit(0)
  #=============================================================
  def reset(self, sizes):
    """"""

    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return

  #=============================================================
  def rebucket(self):
    """"""

    buff = self._file_iterator.next()
    len_cntr = Counter()

    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)

    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return

  #=============================================================
  def _finalize(self):
    """"""

    self._metabucket._finalize()
    return

  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, forest_idxs=None, shuffle=True):
    """"""

    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      if batch_size == 0:
        n_splits = 1
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append((bkt_idx, bkt_mb))
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      feed_dict = {}
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]




      maxlen = np.max(np.sum(np.greater(data[:, :, 0], 0), axis=1))
      feed_dict.update({
        self.inputs: data[:, :maxlen, input_idxs],
        self.targets: data[:, :maxlen, target_idxs]
      })
      if self.use_forest and forest_idxs is not None:

        in_neighbor_data = self[bkt_idx].in_neighbor_data[bkt_mb]
        in_neighbor_rel_data = self[bkt_idx].in_neighbor_rel_data[bkt_mb]
        in_neighbor_mask= self[bkt_idx].in_neighbor_mask[bkt_mb]

        out_neighbor_data = self[bkt_idx].out_neighbor_data[bkt_mb]
        out_neighbor_rel_data = self[bkt_idx].out_neighbor_rel_data[bkt_mb]
        out_neighbor_mask = self[bkt_idx].out_neighbor_mask[bkt_mb]
        feed_dict.update({
          self.in_neighbor: in_neighbor_data[:, :maxlen],
          self.in_neighbor_rel: in_neighbor_rel_data[:, :maxlen],
          self.in_neighbor_mask: in_neighbor_mask[:, :maxlen],

          self.out_neighbor: out_neighbor_data[:, :maxlen],
          self.out_neighbor_rel: out_neighbor_rel_data[:, :maxlen],
          self.out_neighbor_mask: out_neighbor_mask[:, :maxlen],
        })
      yield feed_dict, sents

  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts

  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
