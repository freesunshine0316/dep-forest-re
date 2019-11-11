
import sys, os
import numpy
from gensim.models import KeyedVectors

vocab = set(line.strip().split()[0] for line in open(sys.argv[1], 'rU'))
print 'len(vocab)', len(vocab)

f = open(sys.argv[2], 'w')
embeddings = KeyedVectors.load_word2vec_format('/home/lsong10/ws/data.embedding/PubMed-and-PMC-w2v.bin', binary=True)
intersect = set()
for word in vocab:
    if word in embeddings:
        vector = ' '.join([str(v) for v in embeddings[word]])
        print >>f, word, vector
        intersect.add(word)
print len(intersect)

for w in vocab - intersect:
    embedding = ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=200)])
    print >>f, w, embedding

f.close()
