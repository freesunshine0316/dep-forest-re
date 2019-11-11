
import os, sys

PUNCT = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

scores = [0.0]*4

for line in open(sys.argv[1], 'rU'):
    line = line.strip().split()
    if line == []:
        continue
    if line[3] in PUNCT:
        continue
    scores[0] += (line[6] == line[8]) and (line[7] == line[9])
    scores[1] += 1.0
    scores[2] += (line[6] == line[8])
    scores[3] += 1.0

print 'LAS:', scores[0]/scores[1], scores[0], scores[1]
print 'UAS:', scores[2]/scores[3], scores[2], scores[3]
