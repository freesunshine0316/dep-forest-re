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

from configurable import Configurable
from lib.linalg import linear
from lib.models.nn import NN

#***************************************************************
class Bucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(Bucket, self).__init__(*args, **kwargs)
    self._size = None
    self._data = None
    self._sents = None
    return
  
  #=============================================================
  def reset(self, size, pad=False):
    """"""
    
    self._size = size
    if pad:
      self._data = [(0,)]
      self._sents = [('',)]
    else:
      self._data = []
      self._sents = []
    return
  
  #=============================================================
  def add(self, sent):
    """"""
    
    if isinstance(self._data, np.ndarray):
      raise TypeError("The buckets have already been finalized, you can't add more")
    if len(sent) > self.size and self.size != -1:
      raise ValueError('Bucket of size %d received sequence of len %d' % (self.size, len(sent)))
    
    words = [word[0] for word in sent][1:] # remove root
    idxs = [word[1:] for word in sent]
    self._sents.append(words)
    self._data.append(idxs)
    return len(self._data)-1
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self._data is None:
      raise ValueError('You need to reset the Buckets before finalizing them')
    
    if len(self._data) > 0:
      shape = (len(self._data), self.size, 6)
      data = np.zeros(shape, dtype=np.int32)

      if self.use_forest:
        in_neighbor_list = [ [x[6] for x in datum] for datum in self._data]
        in_neighbor_rel_list = [[x[7] for x in datum] for datum in self._data]
        max_in_neighbor_size = max([ max([len(x) for x in y]) for y in in_neighbor_list])
        in_neighbor_data = np.zeros((len(self._data), self.size, max_in_neighbor_size), dtype=np.int32)
        in_neighbor_rel_data = np.zeros((len(self._data), self.size, max_in_neighbor_size), dtype=np.int32)
        in_neighbor_mask = np.zeros((len(self._data), self.size, max_in_neighbor_size), dtype=np.int32)

        out_neighbor_list = [[x[8] for x in datum] for datum in self._data]
        out_neighbor_rel_list = [[x[9] for x in datum] for datum in self._data]
        max_out_neighbor_size = max([max([len(x) for x in y]) for y in out_neighbor_list])
        out_neighbor_data = np.zeros((len(self._data), self.size, max_out_neighbor_size), dtype=np.int32)
        out_neighbor_rel_data = np.zeros((len(self._data), self.size, max_out_neighbor_size), dtype=np.int32)
        out_neighbor_mask = np.zeros((len(self._data), self.size, max_out_neighbor_size), dtype=np.int32)

      for i, datum in enumerate(self._data):

        plain_input_datum = np.array([x[:6] for x in datum])

        data[i, 0:len(datum), :6] = plain_input_datum
        if self.use_forest:
          for j in range(len(datum)):
            in_neighbor_data[i, j, 0: len(in_neighbor_list[i][j])] = np.array(in_neighbor_list[i][j])
            in_neighbor_rel_data[i, j, 0: len(in_neighbor_list[i][j])] = np.array(in_neighbor_rel_list[i][j])
            in_neighbor_mask[i, j, 0: len(in_neighbor_list[i][j])] = 1

            out_neighbor_data[i, j, 0: len(out_neighbor_list[i][j])] = np.array(out_neighbor_list[i][j])
            out_neighbor_rel_data[i, j, 0: len(out_neighbor_list[i][j])] = np.array(out_neighbor_rel_list[i][j])
            out_neighbor_mask[i, j, 0: len(out_neighbor_list[i][j])] = 1
      self._data = data
      self._sents = np.array(self._sents)

      if self.use_forest:
        self._in_neighbor_data = in_neighbor_data
        self._in_neighbor_rel_data = in_neighbor_rel_data
        self._in_neighbor_mask = in_neighbor_mask

        self._out_neighbor_data = out_neighbor_data
        self._out_neighbor_rel_data = out_neighbor_rel_data
        self._out_neighbor_mask = out_neighbor_mask
    else:
      self._data = np.zeros((0, 1), dtype=np.float32)
      self._sents = np.zeros((0, 1), dtype=str)
      if self.use_forest:
        self._in_neighbor_data = np.zeros((0, 1), dtype=np.float32)
        self._in_neighbor_rel_data = np.zeros((0, 1), dtype=np.float32)
        self._in_neighbor_mask = np.zeros((0, 1), dtype=np.float32)

        self._out_neighbor_data = np.zeros((0, 1), dtype=np.float32)
        self._out_neighbor_rel_data = np.zeros((0, 1), dtype=np.float32)
        self._out_neighbor_mask = np.zeros((0, 1), dtype=np.float32)
    print('Bucket %s is %d x %d' % ((self._name,) + self._data.shape[0:2]))
    return
  
  #=============================================================
  def __len__(self):
    return len(self._data)
  
  #=============================================================
  @property
  def size(self):
    return self._size
  @property
  def data(self):
    return self._data
  @property
  def sents(self):
    return self._sents
  @property
  def in_neighbor_data(self):
    return self._in_neighbor_data
  @property
  def in_neighbor_rel_data(self):
    return self._in_neighbor_rel_data
  @property
  def in_neighbor_mask(self):
    return self._in_neighbor_mask
  @property
  def out_neighbor_data(self):
    return self._out_neighbor_data
  @property
  def out_neighbor_rel_data(self):
    return self._out_neighbor_rel_data
  @property
  def out_neighbor_mask(self):
    return self._out_neighbor_mask