import json
import re
import codecs
import numpy as np
import random
import padding_utils
import collections


def path_centric_pruning(triples, e1, e2):
    assert type(e1) == type(e2) == set
    if len(e1 & e2) > 0:
        lca = list(e1 & e2)[0]
    else:
        heads = collections.defaultdict(set)
        for prb,mi,hi,lb in triples:
            heads[mi].add(hi)
        e1_queue = collections.deque(e1)
        e2_queue = collections.deque(e2)
        lca = None
        while len(e1_queue) > 0 or len(e2_queue) > 0:
            if len(e1_queue) > 0:
                i1 = e1_queue.popleft()
                for h1 in heads[i1]:
                    if h1 >= 0 and h1 not in e1:
                        e1.add(h1)
                        e1_queue.append(h1)
                        if h1 in e2:
                            lca = h1
                            break
            if lca != None:
                break
            if len(e2_queue) > 0:
                i2 = e2_queue.popleft()
                for h2 in heads[i2]:
                    if h2 >= 0 and h2 not in e2:
                        e2.add(h2)
                        e2_queue.append(h2)
                        if h2 in e1:
                            lca = h2
                            break
            if lca != None:
                break
    if lca == None: # can't find a legal path!
        print('!!! No legal path, use full tree')
        return
    return e1 | e2


def read_1best_file(inpath, options):
    # First, read heads and dependency_relations
    all_heads = [[]]
    all_deprels = [[]]
    for line in codecs.open(inpath, 'rU', 'utf-8'):
        line = line.strip()
        if line == '':
            all_heads.append([])
            all_deprels.append([])
        else:
            line = line.split()
            all_heads[-1].append(int(line[6]))
            all_deprels[-1].append(line[7])
    if len(all_heads[-1]) == 0:
        all_heads.pop()
        all_deprels.pop()
    ##return heads, deprel

    # Second, make triples
    all_triples = []
    for heads, deprels in zip(all_heads, all_deprels):
        triples = []
        for i, (h,l) in enumerate(zip(heads, deprels)):
            h = int(h)-1
            triples.append((None,i,h,l)) # (prb, mi, hi, lb)
        all_triples.append(triples)
    return all_triples


def read_cubesparse_file(inpath, options):
    all_triples = []
    all_cubesparse, all_sents, _ = json.load(codecs.open(inpath, 'r', 'utf-8'))
    for i in range(len(all_cubesparse)):
        triples = []
        for (prb,mi,hi,lb) in all_cubesparse[i]:
            assert mi >= 1 and hi >= 0
            if prb >= options.forest_edge_thres:
                triples.append((prb,mi-1,hi-1,lb))
        all_triples.append(triples)
    return all_triples


def read_nbest_file(inpath, options):
    all_triples = []
    all_nbest, all_sents, _ = json.load(codecs.open(inpath, 'r', 'utf-8'))
    for i in range(len(all_nbest)):
        triples = {}
        for j, hypo in enumerate(all_nbest[i]):
            if j >= options.forest_nbest_thres:
                break
            for prb,mi,hi,lb in hypo:
                assert 0.0 <= prb and prb <= 1.0
                if (mi,hi,lb) not in triples:
                    triples[(mi,hi,lb)] = prb
                else:
                    triples[(mi,hi,lb)] = max(triples[(mi,hi,lb)], prb)
        new_triples = []
        for k, prb in triples.iteritems():
            mi, hi, lb = k
            assert mi >= 1 and hi >= 0
            new_triples.append((prb,mi-1,hi-1,lb))
        all_triples.append(new_triples)
    return all_triples


def triples_to_adjacent_list(sent_len, triples, i_keep, options):
    in_neigh = [[i,] for i in range(sent_len)]
    in_neigh_rel = [['self',] for i in range(sent_len)]
    out_neigh = [[i,] for i in range(sent_len)]
    out_neigh_rel = [['self',] for i in range(sent_len)]
    if options.forest_prob_aware and options.forest_type != '1best':
        in_neigh_prob = [[1.0,] for i in range(sent_len)]
        out_neigh_prob = [[1.0,] for i in range(sent_len)]
    else:
        in_neigh_prob = None
        out_neigh_prob = None
    for prb,mi,hi,lb in triples:
        assert mi >= 0 and hi >= -1
        if mi not in i_keep and hi not in i_keep:
            continue
        if hi < 0:
            in_neigh[mi].append(mi)
            in_neigh_rel[mi].append(lb) # root
            if in_neigh_prob != None:
                in_neigh_prob[mi].append(1.0)
        else:
            in_neigh[mi].append(hi)
            in_neigh_rel[mi].append(lb) # regular
            out_neigh[hi].append(mi)
            out_neigh_rel[hi].append(lb+'-rev')
            if in_neigh_prob != None:
                in_neigh_prob[mi].append(prb)
                out_neigh_prob[hi].append(prb)

    return in_neigh, in_neigh_rel, in_neigh_prob, out_neigh, out_neigh_rel, out_neigh_prob


def read_bionlp_file(inpath, inpath_dep, options):
    refmap = {None:0, 'CPR:3':1, 'CPR:4':2, 'CPR:5':3, 'CPR:6':4, 'CPR:9':5,}
    nemap = {None:0, 'CHEMICAL-B':1, 'CHEMICAL-I':2, 'GENE-Y-B':3, 'GENE-Y-I':4, 'GENE-N-B':5, 'GENE-N-I':6,}

    options.num_relations = len(refmap)
    options.num_nes = len(set(nemap.values()))

    if options.forest_type == '1best':
        all_triples = read_1best_file(inpath_dep, options)
    elif options.forest_type == 'cubesparse':
        all_triples = read_cubesparse_file(inpath_dep, options)
    elif options.forest_type == 'nbest':
        all_triples = read_nbest_file(inpath_dep, options)
    else:
        assert False, 'not supported'

    all_ids = [] # [batch]
    all_toks = [] # [batch, sentence]
    all_poses = [] # [batch, sentence]
    all_nes = [] # [batch, sentence]
    all_in_neigh = [] # [batch, sentence, neigh]
    all_in_label = [] # [batch, sentence, neigh]
    all_in_prob = [] # [batch, sentence, neigh]
    all_out_neigh = [] # [batch, sentence, neigh]
    all_out_label = [] # [batch, sentence, neigh]
    all_out_prob = [] # [batch, sentence, neigh]
    all_entity_indices = [] # [batch, 2, indices]
    all_refs = [] # [batch]
    total, inthres, outthres = 0.0, 0.0, 0.0
    max_sentlen, max_indices = 0, 0
    g = 0
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for file_data in json.load(f):
            file_id = file_data['file_id']
            for inst in file_data['file_data']:
                N = len(inst['toks'])
                if options.case == False:
                    inst['toks'] = [x.lower() for x in inst['toks']]
                if len(inst['id'].split()) == 2:
                    all_ids.append(file_id + ' ' + inst['id'])
                else:
                    all_ids.append(inst['id'])
                all_toks.append(inst['toks'])
                all_poses.append(inst['poses'])
                all_nes.append([0 for i in range(N)])
                si, sj, ei, ej = inst['subj_start'], inst['subj_end'], inst['obj_start'], inst['obj_end']
                all_nes[-1][si] = nemap[inst['subj_type']+'-B']
                for i in range(si+1, sj):
                    all_nes[-1][i] = nemap[inst['subj_type']+'-I']
                all_nes[-1][ei] = nemap[inst['obj_type']+'-B']
                for i in range(ei+1, ej):
                    all_nes[-1][i] = nemap[inst['obj_type']+'-I']
                all_entity_indices.append([range(si,sj), range(ei,ej)])
                all_refs.append(refmap[inst['ref']])

                i_keep = set(range(N))
                if hasattr(options, 'path_centric_pruning') and options.path_centric_pruning:
                    i_keep = path_centric_pruning(all_triples[g], set(range(si,sj)), set(range(ei,ej)))
                    if i_keep == None:
                        i_keep = set(range(N))

                in_neigh, in_label, in_prob, out_neigh, out_label, out_prob = triples_to_adjacent_list(N,
                        all_triples[g], i_keep, options)

                max_sentlen = max(max_sentlen, N)
                for inn, outn in zip(in_neigh, out_neigh):
                    total += 1.0
                    inthres += 1.0 if len(inn) > options.max_in_neigh_num else 0.0
                    outthres += 1.0 if len(outn) > options.max_out_neigh_num else 0.0

                all_in_neigh.append(in_neigh)
                all_in_label.append(in_label)
                all_in_prob.append(in_prob)
                all_out_neigh.append(out_neigh)
                all_out_label.append(out_label)
                all_out_prob.append(out_prob)
                g += 1

    print('{} cases exceed the in_neigh_thres {}'.format(inthres/total, options.max_in_neigh_num))
    print('{} cases exceed the out_neigh_thres {}'.format(outthres/total, options.max_out_neigh_num))
    print('Maximal sentence length: {}'.format(max_sentlen))
    print('Positive percent: {}'.format(100.0*sum([x != 0 for x in all_refs])/len(all_refs)))
    return zip(all_toks, all_poses, all_nes, all_entity_indices, \
            all_in_neigh, all_in_label, all_in_prob, all_out_neigh, all_out_label, all_out_prob, all_refs, all_ids)


def collect_vocabs(all_instances, all_words, all_chars, all_poses, all_edgelabels):
    for instance in all_instances:
        toks, poses, in_labels, out_labels = instance[0], instance[1], instance[5], instance[8]
        all_words.update(toks)
        all_poses.update(poses)
        for il in in_labels: all_edgelabels.update(il)
        for ol in out_labels: all_edgelabels.update(ol)
    for w in all_words:
        all_chars.update(w)


class G2SDataStream(object):
    def __init__(self, options, all_instances, word_vocab, char_vocab, pos_vocab, edgelabel_vocab,
                 isShuffle=False, isLoop=False, isSort=True, is_training=False):
        self.options = options
        batch_size = options.batch_size
        # index tokens and filter the dataset
        processed_instances = []
        unk_count, total_count = 0.0, 0.0
        unk_idx = word_vocab.getIndex('UNK')
        for (toks, poses, nes, entity_indices, in_neigh, in_label, in_prob, out_neigh, out_label, out_prob,
                ref, id) in all_instances:
            in_neigh = [x[:options.max_in_neigh_num] for x in in_neigh]
            in_label = [x[:options.max_in_neigh_num] for x in in_label]
            out_neigh = [x[:options.max_out_neigh_num] for x in out_neigh]
            out_label = [x[:options.max_out_neigh_num] for x in out_label]
            if in_prob != None:
                in_prob = [x[:options.max_in_neigh_num] for x in in_prob]
                out_prob = [x[:options.max_out_neigh_num] for x in out_prob]

            toks_idx = word_vocab.to_index_sequence_for_list(toks)
            unk_count += sum([x == unk_idx for x in toks_idx])
            total_count += len(toks_idx)

            if is_training and options.word_dropout_rate > 0.0:
                for i in range(len(toks_idx)):
                    if random.random() < options.word_dropout_rate:
                        toks_idx[i] = unk_idx

            toks_chars_idx = None
            if options.with_char:
                toks_chars_idx = char_vocab.to_character_matrix_for_list(toks,
                        max_char_per_word=options.max_char_per_word)

            poses_idx = None
            if options.with_POS:
                poses_idx = word_vocab.to_index_sequence_for_list(poses)

            in_label_idx = [edgelabel_vocab.to_index_sequence_for_list(il) for il in in_label]
            out_label_idx = [edgelabel_vocab.to_index_sequence_for_list(ol) for ol in out_label]

            processed_instances.append((toks_idx, toks_chars_idx, poses_idx, nes, entity_indices,
                in_neigh, in_label_idx, in_prob, out_neigh, out_label_idx, out_prob, ref, id))

        print('UNK percent {}'.format(unk_count/total_count))
        all_instances = processed_instances

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda inst: len(inst[0]))
        elif isShuffle:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = G2SBatch(cur_instances, options, word_vocab=word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class G2SBatch(object):
    def __init__(self, instances, options, word_vocab=None):
        self.options = options

        self.instances = instances # list of tuples
        self.batch_size = len(instances)
        self.vocab = word_vocab

        # sentence length
        self.sentence_lengths = [] # [batch_size]
        for inst in instances:
            self.sentence_lengths.append(len(inst[0]))
        self.sentence_lengths = np.array(self.sentence_lengths, dtype=np.int32)

        # sentence char length
        if options.with_char:
            self.sentence_chars_lengths = [[len(toks_chars_idx) for toks_chars_idx in inst[1]] for inst in instances]
            self.sentence_chars_lengths = padding_utils.pad_2d_vals_no_size(self.sentence_chars_lengths)

#(0-toks_idx, 1-toks_chars_idx, 2-poses_idx, 3-nes, 4-entity_indices,
#                5-in_neigh, 6-in_label_idx, 7-in_prob, 8-out_neigh, 9-out_label_idx, 10-out_prob, 11-ref, 12-id)
        # neigh mask
        self.in_neigh_mask = [] # [batch_size, sentence_num, neigh_num]
        self.out_neigh_mask = [] # [batch_size, sentence_num, neigh_num]
        self.entity_indices_mask = [] # [batch_size, 2, indices]
        for inst in instances:
            eee = [[1 for x in entity] for entity in inst[4]]
            self.entity_indices_mask.append(eee)
            iii = [[1 for x in in_neigh] for in_neigh in inst[5]]
            self.in_neigh_mask.append(iii)
            ooo = [[1 for x in out_neigh] for out_neigh in inst[8]]
            self.out_neigh_mask.append(ooo)
        self.in_neigh_mask = padding_utils.pad_3d_vals_no_size(self.in_neigh_mask)
        self.out_neigh_mask = padding_utils.pad_3d_vals_no_size(self.out_neigh_mask)
        self.entity_indices_mask = padding_utils.pad_3d_vals_no_size(self.entity_indices_mask)

        # the actual contents
        self.sentence_words = [x[0] for x in instances]
        if options.with_char:
            self.sentence_chars = [x[1] for x in instances] # [batch_size, sent_len, char_num]
        if options.with_POS:
            self.sentence_POSs = [x[2] for x in instances]
        self.nes = [x[3] for x in instances] # [batch_size, sent_len]
        self.entity_indices = [x[4] for x in instances] # [batch_size, 2, indices]
        self.in_neigh_indices = [x[5] for x in instances]
        self.in_neigh_edges = [x[6] for x in instances]
        self.out_neigh_indices = [x[8] for x in instances]
        self.out_neigh_edges = [x[9] for x in instances]
        if instances[0][7] != None:
            self.in_neigh_prob = [x[7] for x in instances]
            self.out_neigh_prob = [x[10] for x in instances]
        self.refs = [x[11] for x in instances]
        self.ids = [x[12] for x in instances]

        # making ndarray
        self.sentence_words = padding_utils.pad_2d_vals_no_size(self.sentence_words)
        if options.with_char:
            self.sentence_chars = padding_utils.pad_3d_vals_no_size(self.sentence_chars)
        if options.with_POS:
            self.sentence_POSs = padding_utils.pad_2d_vals_no_size(self.sentence_POSs)
        self.nes = padding_utils.pad_2d_vals_no_size(self.nes)
        self.entity_indices = padding_utils.pad_3d_vals_no_size(self.entity_indices)
        self.in_neigh_indices = padding_utils.pad_3d_vals_no_size(self.in_neigh_indices)
        self.in_neigh_edges = padding_utils.pad_3d_vals_no_size(self.in_neigh_edges)
        self.out_neigh_indices = padding_utils.pad_3d_vals_no_size(self.out_neigh_indices)
        self.out_neigh_edges = padding_utils.pad_3d_vals_no_size(self.out_neigh_edges)
        if instances[0][7] != None:
            self.in_neigh_prob = padding_utils.pad_3d_vals_no_size(self.in_neigh_prob)
            self.out_neigh_prob = padding_utils.pad_3d_vals_no_size(self.out_neigh_prob)
        self.refs = np.asarray(self.refs, dtype='int32')

        assert self.in_neigh_mask.shape == self.in_neigh_indices.shape
        assert self.in_neigh_mask.shape == self.in_neigh_edges.shape
        assert self.out_neigh_mask.shape == self.out_neigh_indices.shape
        assert self.out_neigh_mask.shape == self.out_neigh_edges.shape


if __name__ == "__main__":
    all_instances = read_bionlp_file('./data/dev.json', options)

