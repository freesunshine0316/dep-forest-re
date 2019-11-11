
import os, sys, heapq, copy
import numpy as np
from collections import defaultdict
from vocab import Vocab
import time

NBEST = 10

class Hypo:
    def __init__(self, logp, edges, u, num_roots):
        self.logp = logp
        self.edges = edges
        self.u = u
        self.num_roots = num_roots

    def __str__(self):
        pass


def cube_next(lhs_list, rhs_list, visited, priq,
        is_making_incomplete, u, k1, k2, new_uas, new_las, is_s_0 = False):
    if len(lhs_list) <= k1 or len(rhs_list) <= k2 or \
            (u, k1, k2) in visited:
        return
    visited.add((u,k1,k2))

    uas_logp = lhs_list[k1].logp + rhs_list[k2].logp
    if is_making_incomplete: # making incomplete hypothesis, adding an edge
        uas_logp += new_uas
        if is_s_0: # s == 0 and is making ('->', 0), must have ROOT relation
            las_logp = uas_logp + new_las[Vocab.ROOT]
            heapq.heappush(priq, (-las_logp,u,k1,k2,Vocab.ROOT))
        else:
            for i, logp in enumerate(new_las):
                if i not in (Vocab.PAD, Vocab.ROOT, Vocab.UNK):
                    las_logp = uas_logp + logp
                    heapq.heappush(priq, (-las_logp,u,k1,k2,i))
    else:
        heapq.heappush(priq, (-uas_logp,u,k1,k2,None))


def cube_pruning(s, t, kk, memory, parse_probs, rel_probs):
    if s == 0 and kk[0] == '<-': # artificial root can't be governed
        return

    key = (s,t) + kk

    hd, md = (s,t) if kk[0] == '->' else (t,s)
    new_uas = np.log(parse_probs[md,hd]+1e-10)
    new_las = np.log(rel_probs[md,hd,:]+1e-10)

    if kk[1] == 0:
        u_range = range(s,t)
        u_inc = 1
        ll, rr = ('->',1), ('<-',1)
    elif kk[1] == 1 and kk[0] == '<-':
        u_range = range(s,t)
        u_inc = 0
        ll, rr = ('<-',1), ('<-',0)
    else:
        u_range = range(s+1,t+1)
        u_inc = 0
        ll, rr = ('->',0), ('->',1)

    #print 'cube_pruning:', key, ll, rr

    ## initialize priority queue
    priq = []
    visited = set() # each item is (split_u, k1, k2)
    for u in u_range:
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, 0, 0, new_uas, new_las, s==0)

    ## actual cube pruning
    nbest = []
    while len(priq) > 0:
        ### obtain the current best
        neglogp, u, k1, k2, li = heapq.heappop(priq)
        logp = -neglogp
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        edges = memory[lhs][k1].edges | memory[rhs][k2].edges
        num_roots = memory[lhs][k1].num_roots + memory[rhs][k2].num_roots
        if li is not None:
            edges.add((md,hd,li))
            num_roots += (s == 0)
        ### check if violates
        is_violate = (num_roots > 1)
        j = -1
        for i, hyp in enumerate(nbest):
            #### hypotheses with same edges should have same logp
            if is_violate or hyp.edges == edges: ##or \
                    ##(i == 0 and hyp.logp - logp >= 10.0):
                is_violate = True
                break
            if hyp.logp < logp:
                j = i
                break
        ### insert
        if not is_violate :
            new_hyp = Hypo(logp, edges, u, num_roots)
            if j == -1:
                nbest.append(new_hyp)
            else:
                nbest.insert(j, new_hyp)
        if len(nbest) >= NBEST:
            break
        ### append new to priq
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1+1, k2, new_uas, new_las, s==0)
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1, k2+1, new_uas, new_las, s==0)
    memory[key] = nbest[:NBEST]


def eisner_dp_nbest(length, parse_probs, rel_probs):
    st_time = time.time()
    memory = defaultdict(list)
    for i in range(0, length+1):
        for d in ('->', '<-'):
            for c in range(2):
                memory[(i,i,d,c)].append(Hypo(0.0, set(), None, 0))

    for t in range(1, length+1):
        for s in range(t-1, -1, -1):
            cube_pruning(s, t, ('<-',0), memory, parse_probs, rel_probs)
            cube_pruning(s, t, ('->',0), memory, parse_probs, rel_probs)
            cube_pruning(s, t, ('<-',1), memory, parse_probs, rel_probs)
            cube_pruning(s, t, ('->',1), memory, parse_probs, rel_probs)
    ## output nbest of memory[(0,length,'->',1)]
    #for hyp in memory[(0,length,'->',1)]:
    #    print hyp.edges, hyp.logp, hyp.num_roots
    print 'Length %d, time %f' %(length, time.time()-st_time)

    #return [list(hyp.edges) for hyp in memory[(0,length,'->',1)]] # return edges containing (mi,hi,lb)
    nbest = []
    for hyp in memory[(0,length,'->',1)]:
        nbest.append([])
        for mi,hi,lb in hyp.edges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((prb,mi,hi,lb))
    return nbest


if __name__ == '__main__':
    # do some unit test
    Vocab.ROOT = 0
    parse_probs = np.arange(1, 10001, dtype=np.float32).reshape((100,100))
    parse_probs = parse_probs/np.sum(parse_probs, axis=-1, keepdims=True)
    rel_probs = np.arange(1, 300001, dtype=np.float32).reshape((100,100,30))
    rel_probs = rel_probs/np.sum(rel_probs, axis=-1, keepdims=True)
    #print parse_probs
    #print rel_probs
    eisner_dp_nbest(99, parse_probs, rel_probs)
    #print Vocab.ROOT

