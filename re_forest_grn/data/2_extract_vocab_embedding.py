
import numpy
import sys, os, codecs
from collections import Counter

vocab = set(line.strip().split()[0] for line in codecs.open(sys.argv[1], 'rU', 'utf-8'))
print 'len(vocab)', len(vocab)

vocab_whole = [line.strip()
        for line in codecs.open('/home/lsong10/ws/data.embedding/bioasq.pubmed.vocab', 'rU', 'utf-8')]

f = codecs.open(sys.argv[2], 'w', 'utf-8')
intersect = set()
for i, line in enumerate(open('/home/lsong10/ws/data.embedding/bioasq.pubmed.200d.txt', 'rU')):
    word = vocab_whole[i]
    if word not in intersect and word in vocab:
        print >>f, word, line.strip()
        intersect.add(word)
print len(intersect)

for w in vocab - intersect:
    embedding = ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=200)])
    print >>f, w, embedding

f.close()
