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

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN
#from lib.models.parsers import eisner_dp_nbest
from lib.models.parsers.eisner_nbest import eisner_dp_nbest

import math

#***************************************************************
class BaseParser(NN):
  """"""

  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""

    raise NotImplementedError

  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""

    raise NotImplementedError

  #=============================================================
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject, feed_dict={}):
    """"""

    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 0:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = vocabs[1][gold[0]]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          gold_parse = gold[1]
          gold_rel = vocabs[2][gold[2]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return

  #=============================================================
  def validate_cube(self, mb_inputs, mb_targets, mb_probs, is_sparse, rel_vocab):
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs # mb_parse_probs: [batch, words, words], mb_rel_probs: [batch, words, words, rel]
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      assert parse_probs.shape[0] >= length + 1
      assert parse_probs.shape == rel_probs.shape[:2]
      if parse_probs.shape[0] > length + 1:
        parse_probs = parse_probs[:length+1,:length+1]
        rel_probs = rel_probs[:length+1,:length+1,:]
      final_probs = np.expand_dims(parse_probs, axis=2) * rel_probs # [words, words, rel]
      if is_sparse:
        triples = []
        for mi in range(1, final_probs.shape[0]):
          for hi in range(final_probs.shape[1]):
            for lb in range(final_probs.shape[2]):
              if final_probs[mi,hi,lb] >= 0.01:
                triples.append((float(final_probs[mi,hi,lb]),mi,hi,rel_vocab[lb]))
        assert len(triples) > 0
        sents.append(triples) # [batch, triples, prb-mi-hi-lb]
      else:
        assert False, 'Under construction. We should also output real relation strings!'
        sents.append(final_probs.tolist()) # [batch, words, words, rel]
    return sents

  def validate_nbest(self, mb_inputs, mb_targets, mb_probs, rel_vocab):
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      nbest = [[[float(x[0]), int(x[1]), int(x[2]), rel_vocab[x[-1]],] for x in cur_best] \
              for cur_best in eisner_dp_nbest(length, parse_probs, rel_probs)]
      sents.append(nbest) # [batch, nbest, edges]
    return sents

  def validate(self, mb_inputs, mb_targets, mb_probs):
    """"""

    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)

      sent = -np.ones( (length, 9), dtype=int)
      tokens = np.arange(1, length+1)
      sent[:,0] = tokens
      sent[:,1:4] = inputs[tokens]
      sent[:,4] = targets[tokens,0]
      sent[:,5] = parse_preds[tokens]
      sent[:,6] = rel_preds[tokens]
      sent[:,7:] = targets[tokens, 1:]
      sents.append(sent)
    return sents

  #=============================================================
  @staticmethod
  def evaluate(filename, punct=NN.PUNCT):
    """"""

    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
    correct = {k:np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2)
  @property
  def target_idxs(self):
    return (3, 4, 5)

  @property
  def forest_idxs(self):
    return (6, 7, 8, 9)
