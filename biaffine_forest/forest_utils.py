
import os, sys, json, codecs
import heapq
import numpy as np
from collections import defaultdict


def triples_to_adjacent_list(sent_len, triples, prb_form=None):
  in_neigh = [[] for i in range(sent_len+1)]
  in_neigh_rel = [[] for i in range(sent_len+1)]
  out_neigh = [[] for i in range(sent_len+1)]
  out_neigh_rel = [[] for i in range(sent_len+1)]
  for prb, mi, hi, lb in triples:
    assert mi <= sent_len and hi <= sent_len
    if prb_form != None:
      pass # TODO: how to change the form of probabilities
    in_neigh[mi].append(hi)
    in_neigh_rel[mi].append(lb)
    out_neigh[hi].append(mi)
    out_neigh_rel[hi].append(lb)
  return (in_neigh, in_neigh_rel, out_neigh, out_neigh_rel)


## triple size: 4 (prb, mi, hi, lb)
def nbest_to_adjacent_list(sent_len, nbest, only_keep):
  assert all(len(h) == sent_len for h in nbest)
  assert all(len(e) == 4 for e in nbest[0])
  assert type(nbest[0][0]) == tuple
  triples = set()
  for i in range(len(nbest)):
    if i >= only_keep > 0:
      break
    triples.update(nbest[i])
  triples = list(triples)
  return triples, triples_to_adjacent_list(sent_len, triples)


## triple size: 4 (prb, mi, hi, lb)
def cube_to_adjacent_list(sent_len, cube, only_keep, ori_rel_vocab, rel_vocab):
  assert False, 'TODO: map ori_lb to lb with ori_rel_vocab and rel_vocab'
  assert len(cube.shape) == 3 # [mi, hi, lb]
  assert cube.shape[0] == cube.shape[1] and cube.shape[0] == sent_len + 1
  triples = []
  for mi in range(1, cube.shape[0]):
    keep = []
    for hi in range(cube.shape[1]):
      for lb in range(cube.shape[2]):
        if type(only_keep) == int:
          heapq.heappush(keep, (cube[mi,hi,lb],mi,hi,lb))
          if len(keep) > only_keep: # such as when only_keep ==1 and len(keep) == 2
            heapq.heappop(keep)
        else:
          if cube[mi,hi,lb] >= only_keep:
            keep.append((cube[mi,hi,lb],mi,hi,lb))
    triples.extend(keep)
  return triples, triples_to_adjacent_list(sent_len, triples)


def load_cube(path, only_keep, rel_vocab):
  if type(only_keep) == int:
    assert only_keep >= 1
  else:
    assert 0.0 < only_keep and only_keep < 1.0
  mapping = defaultdict(dict)
  total_words = 0.0
  total_edges = 0.0
  with codecs.open(path, 'r', 'utf-8') as f:
    cubes, sents, ori_rel_vocab = json.load(f)
    for i, bkt in enumerate(sents):
      for j, sent in enumerate(bkt):
        sent = tuple(sent)
        cube = np.array(cubes[i][j])
        triples, adj_lists = cube_to_adjacent_list(len(sent), cube, only_keep)
        mapping[len(sent)][sent] = (triples, adj_lists)
        total_words += len(sent)
        total_edges += len(triples)
  print sum(len(d.keys()) for d in mapping.values())
  print 'On average each word has %f edges' % (total_edges/total_words)
  return mapping


def load_cubesparse(path, only_keep, rel_vocab):
  assert type(only_keep) == float and 0.0 < only_keep and only_keep < 1.0
  mapping = []
  total_words = 0.0
  total_edges = 0.0
  with codecs.open(path, 'r', 'utf-8') as f:
    cubesparse, sents, _ = json.load(f)
    for i, sent in enumerate(sents):
      sent = tuple([x.encode('utf-8') for x in sent])
      triples = []
      for (prb,mi,hi,lb) in cubesparse[i]:
        if prb >= only_keep:
          lb = rel_vocab[lb]
          triples.append((prb,mi,hi,lb))
      adj_lists = triples_to_adjacent_list(len(sent), triples)
      mapping.append((triples, adj_lists))
      total_words += len(sent)
      total_edges += len(triples)
  print len(mapping)
  print 'On average each word has %f edges' % (total_edges/total_words)
  return mapping


def load_nbest(path, only_keep, rel_vocab):
  assert type(only_keep) == int and only_keep >= 1
  mapping = []
  total_words = 0.0
  total_edges = 0.0
  with codecs.open(path, 'r', 'utf-8') as f:
    nbests, sents, _ = json.load(f)
    for i, sent in enumerate(sents):
      sent = tuple([x.encode('utf-8') for x in sent])
      nbest = [[tuple(y[:-1]+[rel_vocab[y[-1]],]) for y in x] for x in nbests[i]]
      triples, adj_lists = nbest_to_adjacent_list(len(sent), nbest, only_keep)
      mapping.append((triples, adj_lists))
      total_words += len(sent)
      total_edges += len(triples)
  print len(mapping)
  print 'On average each word has %f edges' % (total_edges/total_words)
  return mapping


def evalb(ans, ref, scores):
  ans = set(x[1:] for x in ans)
  ans_uas = set(x[:2] for x in ans)
  for mi, hi, lb in ref:
    scores[0] += (mi, hi, lb) in ans
    scores[1] += 1.0
    scores[2] += (mi, hi) in ans_uas
    scores[3] += 1.0


if __name__ == '__main__':
  # load rel vocab
  rel_vocab = {'pad':0, 'root':1, 'unk':2}
  with codecs.open('saves/ptb/rels.txt', 'rU', 'utf-8') as f:
    for line in f:
      rel = line.strip().split()[0]
      if rel not in rel_vocab:
        i = len(rel_vocab)
        rel_vocab[rel] = i

  #mapping = load_cubesparse('saves/ptb_cham_prot/train.conllu_cubesparse.json', 0.2, rel_vocab)
  #mapping = load_nbest('saves/ptb_cham_prot/dev.conllu_nbest.json', 2, rel_vocab)

  # load data
  mapping = load_cubesparse('saves/ptb/genia_train.conllu_100_cubesparse.json', 0.05, rel_vocab)
  #mapping = load_nbest('saves/ptb/genia_train.conllu_100_nbest.json', 1, rel_vocab)

  # check accuracy
  scores = [0.0, ] * 4 # [las_right, las_total, uas_right, uas_total]
  sent = []
  triples_ref = []
  n = 0
  with codecs.open('data/ptb/genia_train.conllu_100', 'rU', 'utf-8') as f:
    for line in f:
      line = line.strip().split()
      if line == []:
        if len(sent) > 0:
          triples_ans, _ = mapping[n]
          evalb(triples_ans, triples_ref, scores)
          sent = []
          triples_ref = []
          n += 1
      else:
        sent.append(line[1])
        lb = line[7]
        if lb not in rel_vocab:
          print 'OOV relation:', lb
          lb = 2
        else:
          lb = rel_vocab[lb]
        mi, hi = int(line[0]), int(line[6])
        triples_ref.append((mi, hi, lb))
  print 'LAS:', scores[0]/scores[1], scores[0], scores[1]
  print 'UAS:', scores[2]/scores[3], scores[2], scores[3]
