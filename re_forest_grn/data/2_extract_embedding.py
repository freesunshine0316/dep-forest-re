
import numpy
import sys, os
from collections import Counter

vocab = set(line.strip().split()[0] for line in open(sys.argv[1], 'rU'))
print 'len(vocab)', len(vocab)

f = open(sys.argv[2], 'w')
intersect = set()
for line in open('/home/lsong10/ws/data.embedding/glove.6B.100d.txt', 'rU'):
    word = line.strip().split()[0]
    if word in vocab:
        print >>f, line.strip()
        intersect.add(word)
print len(intersect)

for w in vocab - intersect:
    embedding = ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=100)])
    print >>f, w, embedding

f.close()
