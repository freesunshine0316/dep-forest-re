# -*- coding: utf-8 -*-

import os, sys, codecs, json
import en_core_web_md
from collections import defaultdict


def read_abstract(path, nlp):
    data = {}
    for line in codecs.open(path, 'rU', 'utf-8'):
        id, title, abstract = line.strip().split('\t')
        doc = doc_ori = title + ' ' + abstract
        doc = doc.replace(',', ' , ', 100)
        doc = doc.replace(':', ' : ', 100)
        doc = doc.replace(';', ' ; ', 100)
        doc = doc.replace('-', ' - ', 100)
        doc = doc.replace('+', ' + ', 100)
        doc = doc.replace('/', ' / ', 100)
        doc = doc.replace('(', ' ( ', 100)
        doc = doc.replace(')', ' ) ', 100)
        doc = doc.replace('<', ' < ', 100)
        doc = doc.replace('>', ' > ', 100)
        doc = doc.replace('[', ' [ ', 100)
        doc = doc.replace(']', ' ] ', 100)
        doc = doc.replace('{', ' { ', 100)
        doc = doc.replace('}', ' } ', 100)
        doc = doc.replace('=', ' = ', 100)
        doc = doc.replace('&', ' & ', 100)
        doc = doc.replace('*', ' * ', 100)
        doc = doc.replace('#', ' # ', 100)
        doc = doc.replace('@', ' @ ', 100)
        doc = doc.replace('CD.', 'CD . ', 100)
        doc = doc.replace(u'·', u' · ', 100)
        doc = doc.replace(u'→', u' → ', 100)
        doc = doc.replace(u'↔', u' ↔ ', 100)
        doc = doc.replace(u'…', u' … ', 100)
        doc = doc.replace(u'↓', u' ↓ ', 100)
        doc = doc.replace(u'β', u' β ', 100)
        doc = doc.replace(u'γ', u' γ ', 100)
        doc = doc.replace(u'Δ', u' Δ ', 100)
        doc = doc.replace(u'', u'  ', 100)
        doc = doc.replace(u'τ', u' τ ', 100)
        doc = doc.replace(u'μg', u' μg ', 100)
        doc = doc.replace('c.', 'c . ', 100)
        doc = doc.replace('i.', 'i . ', 100)
        doc = doc.replace('p.', 'p . ', 100)
        doc = doc.replace('A.', 'A .', 100)
        doc = doc.replace('B.', 'B .', 100)
        doc = doc.replace('C.', 'C .', 100)
        doc = doc.replace('D.', 'D .', 100)
        doc = doc.replace('E.', 'E .', 100)
        doc = doc.replace('F.', 'F .', 100)
        doc = doc.replace('G.', 'G .', 100)
        doc = doc.replace('H.', 'H .', 100)
        doc = doc.replace('I.', 'I .', 100)
        doc = doc.replace('L.', 'L .', 100)
        doc = doc.replace('K.', 'K .', 100)
        doc = doc.replace('P.', 'P .', 100)
        doc = doc.replace('R.', 'R .', 100)
        doc = doc.replace('S.', 'S .', 100)
        doc = doc.replace('T.', 'T .', 100)
        doc = doc.replace('V.', 'V .', 100)
        doc = doc.replace('Y.', 'Y .', 100)
        doc = doc.replace('14KATI', '14 KATI', 100)
        doc = doc.replace('1RNA', '1 RNA', 100)
        doc = doc.replace('His3', 'His 3', 100)
        doc = doc.replace('His4', 'His 4', 100)
        doc = doc.replace('Arg2', 'Arg 2', 100)
        doc = doc.replace('Asp7', 'Asp 7', 100)
        doc = doc.replace('HDACi', 'HDAC i', 100)
        doc = doc.replace('ACEi', 'ACE i', 100)
        doc = doc.replace('CDA70Thr', 'CDA 70Thr', 100)
        doc = doc.replace('DCK24Val', 'DCK 24Val', 100)
        doc = doc.replace('KITD816V', 'KIT D816V', 100)
        doc = doc.replace('Nf1col2', 'Nf1 col2', 100)
        doc = doc.replace('Arg122', 'Arg 122', 100)
        doc = doc.replace('Leu113', 'Leu 113', 100)
        doc = doc.replace('J.Vav1', 'J . Vav1', 100)
        doc = doc.replace('1.2DHP', '1.2 DHP', 100)
        doc = doc.replace('Jun.', 'Jun .', 100)
        doc = doc.replace('aline.', 'aline .', 100)
        doc = doc.replace('talol.', 'talol .', 100)
        doc = doc.replace('mdr1a.', 'mdr1a .', 100)
        doc = doc.replace('nTiO', 'n TiO', 100)
        doc = doc.replace('H3K36me3', 'H3 K36me3', 100)
        doc = doc.replace('MTRA2756G', 'MTR A2756G', 100)
        doc = doc.replace('CLDN3KD', 'CLDN3 KD', 100)
        doc = doc.replace('CLDN4KD', 'CLDN4 KD', 100)
        doc = doc.replace('A431NET', 'A431 NET', 100)
        doc = doc.replace('shTRPC2', 'sh TRPC2', 100)
        doc = doc.replace('ERKO', 'ER KO', 100)
        doc = doc.replace('HSD1', 'HSD 1', 100)
        doc = doc.replace('1 RA', '1 R A', 100)
        doc = doc.replace('Cys', 'Cys ', 100)
        doc = doc.replace('Asn', 'Asn ', 100)
        doc = doc.replace('Met', 'Met ', 100)
        doc = doc.replace('Phe', 'Phe ', 100)
        doc = doc.replace('Glu', 'Glu ', 100)
        doc = doc.replace('Ser', 'Ser ', 100)
        doc = doc.replace('Ala', 'Ala ', 100)
        doc = doc.replace('Gly', 'Gly ', 100)
        doc = doc.replace('Thr', 'Thr ', 100)
        doc = doc.replace('Tyr', 'Tyr ', 100)
        doc = doc.replace('Lys', 'Lys ', 100)
        doc = doc.replace('Lyr', 'Lyr ', 100)
        doc = doc.replace('MTRR', 'MTRR ', 100)
        doc = doc.replace('POX', ' POX ', 100)
        doc = doc.replace('PEC', ' PEC ', 100)
        doc = doc.replace('Trp', ' Trp ', 100)
        doc = doc.replace('MSH', ' MSH ', 100)
        doc = doc.replace('XBP', ' XBP ', 100)
        doc = doc.replace('1629T', ' 1629T', 100)
        doc = ' '.join(doc.split())

        doc_toks = []
        doc_poses = []
        for sent in nlp(doc).sents:
            doc_toks.append([])
            doc_poses.append([])
            for word in sent:
                doc_toks[-1].append(word.text)
                doc_poses[-1].append(word.tag_)

        # (i, j) --> t
        mapping_ij2t = {}
        t = 0
        for i, sent in enumerate(doc_toks):
            for j, word in enumerate(sent):
                mapping_ij2t[(i,j)] = t
                t += 1

        # char index to token index mapping
        mapping_c2st = {}
        mapping_c2ed = {}
        c = 0
        for i, sent in enumerate(doc_toks):
            for j, word in enumerate(sent):
                while doc_ori[c].isspace():
                    c += 1
                mapping_c2st[c] = (i,j)
                for k in range(len(word)):
                    if word[k] != doc_ori[c]:
                        print '%s ||| %s' %(word[k:], paragraph[c:])
                        assert False
                    c += 1
                mapping_c2ed[c] = (i,j)

        data[id] = (doc_ori, doc_toks, doc_poses, mapping_ij2t, mapping_c2st, mapping_c2ed)
    return data


def read_entities(path):
    data = defaultdict(list)
    for line in codecs.open(path, 'rU', 'utf-8'):
        id, tag, tp, char_st, char_ed, txt = line.strip().split('\t')
        data[id].append((tag, tp, int(char_st), int(char_ed), txt))
    return data


def read_reference(path):
    data = defaultdict(dict)
    for line in codecs.open(path, 'rU', 'utf-8'):
        id, rel, e1, e2 = line.strip().split('\t')
        e1 = e1.split(':')[1]
        e2 = e2.split(':')[1]
        data[id][(e1,e2)] = rel
    return data


def process(path_prefix, nlp):
    all_abstract = read_abstract(path_prefix+'_'+'abstracts.tsv', nlp)
    all_entities = read_entities(path_prefix+'_'+'entities.tsv')
    all_reference = read_reference(path_prefix+'_'+'gold_standard.tsv')

    corpus = []
    right, output_total, total = 0.0, 0.0, 0.0
    for id in all_entities.keys():
        doc_ori, doc_toks, doc_poses, mapping_ij2t, mapping_c2st, mapping_c2ed = all_abstract[id]
        reference = all_reference[id]

        new_entities = []
        for tag, tp, char_st, char_ed, txt in all_entities[id]:
            try:
                st = mapping_c2st[char_st]
            except KeyError:
                st = None
                print 'char_st:', doc_ori[char_st-5:char_st+6]
            try:
                ed = mapping_c2ed[char_ed]
            except KeyError:
                ed = None
                print 'char_ed:', doc_ori[char_ed-5:char_ed+6]
            if st == None or ed == None:
                continue
            new_entities.append((tag, tp, st, ed, txt))

        ffile = {'file_data':[], 'file_id':id, }
        for m, (tag1, tp1, st1, ed1, txt1) in enumerate(new_entities):
            if tp1 != 'CHEMICAL':
                continue
            for n, (tag2, tp2, st2, ed2, txt2) in enumerate(new_entities):
                if tp2 == 'CHEMICAL': # second should NOT be chemical
                    continue
                assert st1 != st2 or ed1 != ed2
                min_i = min(st1[0], st2[0])
                max_i = max(ed1[0], ed2[0])
                if max_i - min_i >= 1:
                    continue
                inst = {'toks':[], 'poses':[], }
                for i in range(min_i, max_i+1):
                    inst['toks'] += doc_toks[i]
                    inst['poses'] += doc_poses[i]
                base_idx = mapping_ij2t[(min_i,0)]
                inst['subj_start'] = mapping_ij2t[st1] - base_idx
                inst['subj_end'] = mapping_ij2t[ed1] - base_idx + 1
                inst['subj_type'] = tp1
                inst['obj_start'] = mapping_ij2t[st2] - base_idx
                inst['obj_end'] = mapping_ij2t[ed2] - base_idx + 1
                inst['obj_type'] = tp2

                key = tuple('T%d'%x for x in sorted([int(tag1[1:]),int(tag2[1:]),]))
                print key
                inst['ref'] = reference.get(key, None)
                inst['id'] = ' '.join([tag1,tag2])
                assert max(inst['subj_start'], inst['subj_end'], inst['obj_start'], inst['obj_end']) <= len(inst['toks'])
                ffile['file_data'].append(inst)
        corpus.append(ffile)
        right += sum([x['ref'] != None for x in ffile['file_data']])
        total += len(reference)
        output_total += len(ffile['file_data'])

    print 'Recall', right/total, right, 'Percent', right/output_total, total, output_total
    return corpus


def write_conllu(data, path):
    f = codecs.open(path, 'w', 'utf-8')
    for file_data in data:
        for inst in file_data['file_data']:
            for i, (t, p) in enumerate(zip(inst['toks'], inst['poses'])):
                print >>f, '\t'.join([str(i+1), t, t, p, p, '_', '0', 'root', '_', '_'])
            print >>f, ''
    f.close()


sys.stdout = codecs.getwriter('utf8')(sys.stdout)
nlp = en_core_web_md.load()

data = process('chemprot_development', nlp)
#json.dump(data, codecs.open('dev.json', 'w', 'utf-8'))
#write_conllu(data, 'dev.conllu')

#data = process('chemprot_test', nlp)
#json.dump(data, codecs.open('test.json', 'w', 'utf-8'))
#write_conllu(data, 'test.conllu')

#data = process('chemprot_training', nlp)
#json.dump(data, codecs.open('train.json', 'w', 'utf-8'))
#write_conllu(data, 'train.conllu')

