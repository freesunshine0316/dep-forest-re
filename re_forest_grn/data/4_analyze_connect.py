
import json, os, sys, codecs
from collections import defaultdict, deque


def read_cubesparse_file(inpath, thres):
    all_triples = []
    all_cubesparse, all_sents, _ = json.load(codecs.open(inpath, 'r', 'utf-8'))
    for i in range(len(all_cubesparse)):
        triples = defaultdict(set)
        for (prb,mi,hi,lb) in all_cubesparse[i]:
            assert mi >= 1 and hi >= 0
            if prb >= thres:
                triples[mi-1].add(hi-1)
                triples[hi-1].add(mi-1)
        all_triples.append(triples)
    return all_triples

def search(sbj, obj, relations):
    queue = deque(sbj)
    visited = set(sbj)
    while len(queue) > 0:
        ci = queue.pop()
        for ni in relations.get(ci,set()):
            if ni in obj:
                return 1.0
            if ni not in visited:
                queue.append(ni)
                visited.add(ni)
    return 0.0

def read_bionlp_file(inpath, inpath_dep, thres):
    all_triples = read_cubesparse_file(inpath_dep, thres)
    g = 0
    right, total = 0.0, 0.0
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for file_data in json.load(f):
            file_id = file_data['file_id']
            for inst in file_data['file_data']:
                si, sj, ei, ej = inst['subj_start'], inst['subj_end'], inst['obj_start'], inst['obj_end']
                sbj = range(si,sj)
                obj = set(range(ei,ej))
                right += search(sbj, obj, all_triples[g])
                total += 1.0
                g += 1
    print right/total, right, total

read_bionlp_file('dev.json', 'dev.json_cubesparse', 0.3)
