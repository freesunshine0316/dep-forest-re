
import json
import sys
import re
from collections import Counter
import codecs


def extract(inpath, all_toks):
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for file_data in json.load(f):
            for inst in file_data["file_data"]:
                all_toks.update([x.lower() for x in inst['toks']])


all_toks = Counter()
extract('dev.json', all_toks)
extract('test.json', all_toks)
extract('train.json', all_toks)
all_toks = sorted(all_toks.items(), key=lambda x: -x[1])

f = codecs.open('vocab.txt', 'w', 'utf-8')
for k,v in all_toks:
    print >>f, k, v
f.close()

