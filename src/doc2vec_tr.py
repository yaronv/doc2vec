import gc
import multiprocessing
import os
from random import shuffle

from gensim.models.doc2vec import Doc2Vec

from src.CorpusReader import CorpusReader
from src.DocumentsIterable import DocumentsIterable
from src.config import *
from src.tf_visualizer import TF_visualizer


class Doc2VecTR(object):
    def __init__(self, train_corpus, groups):
        self.train_corpus = train_corpus
        self.groups = groups

    def run(self):
        print('app started')

        cores = multiprocessing.cpu_count()
        print('num of cores is %s' % cores)
        gc.collect()
        if load_existing:
            print('loading an exiting model')
            model = Doc2Vec.load(doc2vec_model)
        else:
            print('reading training corpus from %s' % self.train_corpus)
            train_data = [self.train_corpus]
            if include_groups_in_train:
                train_data += self.groups
            # corpus_data = DocumentsIterable(train_data)
            corpus_data = CorpusReader(train_data).read_corpus()
            model = Doc2Vec(size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)
            print('building vocabulary...')
            model.build_vocab(corpus_data)

            # start training the model
            for epoch in range(epochs):
                print ('Now training epoch %s' % epoch)
                shuffle(corpus_data)
                model.train(corpus_data, total_examples=model.corpus_count, epochs=model.iter)
                # model.alpha -= 0.002  # decrease the learning rate
                # model.min_alpha = model.alpha  # fix the learning rate, no decay

            model.save(doc2vec_model)
            model.save_word2vec_format(word2vec_model)

        print('total docs learned %s' % (len(model.docvecs)))

        groups_vectors = []
        ids = []
        labels = []
        groups_sizes = {}

        # add the groups vectors
        for i, group in enumerate(self.groups):
            print('inferring group of documents from %s' % group)
            group_data = DocumentsIterable([group])
            if i == 0:
                groups_sizes[i] = 0
            else:
                groups_sizes[i] = groups_sizes[i-1]

            for vec in group_data:
                vec_data = model.infer_vector(vec.words)
                groups_vectors.append(vec_data)
                ids.append(vec.tags)
                labels.append(i)
                groups_sizes[i] += 1

        print('writing meta data to file in tensorflow format')
        with open(os.path.join(output_path, meta_file + '.tsv'), 'wb') as file_metadata:
            file_metadata.write(b'doc_id\tgroup' + b'\n')
            for i, id_val in enumerate(ids):
                file_metadata.write((id_val[0] + '\t' + str(groupsNames[labels[i]]) + '\n').encode('utf-8'))

        print('writing vectors to file')
        with open(os.path.join(output_path, meta_file + '-vecs.tsv'), 'wb') as file_metadata:
            for i, vec in enumerate(groups_vectors):
                file_metadata.write((",".join(["{}".format(number) for number in vec]) + '\n').encode('utf-8'))

        # create a new tensor board visualizer
        visualizer = TF_visualizer(groups_sizes, model_dimensions)

        # visualize the data using tensor board
        visualizer.visualize()
