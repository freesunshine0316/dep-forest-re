#import os
#from nltk.parse.stanford import StanfordDependencyParser
#from graphviz import Source
#
#os.environ['CLASSPATH'] = r'/home/lsong10/ws/data.dependency/stanford-parser-full-2018-10-17'
#sentence = 'The brown fox is quick and lazy .'
#
#sdp = StanfordDependencyParser()
#result = list(sdp.raw_parse(sentence))
#result = [x for x in result][0]
#print result
#result = result.to_dot()
#print result
#

import os, sys, codecs, json

path = sys.argv[1]

# (1) get mention indices
indices = []
with codecs.open(path, 'rU', 'utf-8') as f:
    for file_data in json.load(f):
        file_id = file_data['file_id']
        for inst in file_data['file_data']:
            si, sj, ei, ej = inst['subj_start'], inst['subj_end'], inst['obj_start'], inst['obj_end']
            assert si != ei or sj != ej
            indices.append(set(range(si+1,sj+1)))
            indices[-1].update(range(ei+1,ej+1))

# (2) gen dots
f = codecs.open(path+'_dot', 'w', 'utf-8')
prefix = 'edge [dir=forward]\nnode [shape=plaintext]\n\n0 [label="0 (None)"]'

gn = 0
ghead = 'digraph G%d{\n'%gn + prefix
gbody = ''
node_num = 0
for line in codecs.open(path+'_1best', 'rU', 'utf-8'):
    line = line.strip()
    if line == '':
        assert max(indices[gn]) <= node_num
        assert len(indices[gn]) >= 2
        if node_num <= 15:
            print >>f, ghead
        gn += 1
        ghead = 'digraph G%d{\n'%gn + prefix
        if node_num <= 15:
            print >>f, gbody + '}\n'
        gbody = ''
        node_num = 0
    else:
        line = line.split()
        mi, wd, hi, lb = line[0], line[1], line[6], line[7]
        mi, hi = int(mi), int(hi)
        if mi in indices[gn]:
            gbody += '%d [label="%d {%s})"]\n' %(mi, mi, wd)
        else:
            gbody += '%d [label="%d %s"]\n' %(mi, mi, wd)
        gbody += '%d -> %d [label="%s"]\n' %(hi, mi, lb)
        node_num += 1

if gbody != '':
    print >>f, ghead
    print >>f, gbody + '}\n'

f.close()

