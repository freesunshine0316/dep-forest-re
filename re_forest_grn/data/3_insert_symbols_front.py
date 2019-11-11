
import os, sys, json, codecs
import numpy

inpath = sys.argv[1]
outpath = sys.argv[1]+'.st'
f = open(outpath,'w')
i = 0
words = set()
for line in open(inpath,'rU'):
    if i == 0:
        vsize = len(line.strip().split())-1
        print vsize
        print >>f, '\t'.join([str(i), '#pad#', ' '.join([str('%.6f'%x) for x in numpy.zeros(vsize)])])
        i += 1
        print >>f, '\t'.join([str(i), 'UNK', ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=vsize)])])
        i += 1
    line = line.strip().split()
    word = line[0]
    if len(line) != vsize+1 or word in words:
        continue
    words.add(word)
    line = ' '.join(line[1:])
    print >>f, '\t'.join([str(i), word, line])
    i += 1

