import glob
import multiprocessing
import ntpath
import os
import xml.etree.ElementTree as ET
import gensim
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from src.config import *
from tf_visualizer import TF_visualizer


class Doc2VecTR(object):
    def __init__(self, train_corpus, groups):
        self.train_corpus = train_corpus
        self.groups = groups

    def run(self):
        print 'app started'

        cores = multiprocessing.cpu_count()
        print 'num of cores is %s' % (cores)

        if loadExisting:
            print 'loading an exiting model'
            model = Doc2Vec.load(model_path)
            # word_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
        else:
            print 'training a new model'
            corpus_data = list(self.read_corpus(self.train_corpus, -1))

            model = Doc2Vec(size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)

            model.build_vocab(corpus_data)

            # start training
            for epoch in range(epoches):
                print ('Now training epoch %s' % epoch)
                model.train(corpus_data, total_examples=model.corpus_count, epochs=model.iter)
                # model.alpha -= 0.002  # decrease the learning rate
                # model.min_alpha = model.alpha  # fix the learning rate, no decay

            model.save(model_path)
            model.save_word2vec_format(word2vec_path)

        print 'total docs learned %s' % (len(model.docvecs))

        groups_vectors = []
        ids = []
        groups_sizes = []
        size_so_far = -1

        # add the groups vectors
        for group in self.groups:
            group_data = list(self.read_corpus(group, size_so_far))
            size_so_far += len(group_data)
            groups_sizes.append(size_so_far)

            for vec in group_data:
                vec_data = model.infer_vector(vec.words)
                groups_vectors.append(vec_data)
                ids.append(vec.tags)

        print 'writing meta data to file in tensorflow format'
        with open(os.path.join(output_path, meta_file + '.tsv'), 'wb') as file_metadata:
            file_metadata.write('doc_id\tgroup' + '\n')
            for idx, id in enumerate(ids):
                file_metadata.write(id + '\t' + self.get_group_name(idx, groups_sizes) + '\n')

        print 'writing vectors to file'
        with open(os.path.join(output_path, meta_file + '-vecs.tsv'), 'wb') as file_metadata:
            for i, vec in enumerate(groups_vectors):
                file_metadata.write(",".join(["%.15f" % number for number in vec]) + '\n')

        # create a new tensor board visualizer
        visualizer = TF_visualizer(groups_sizes, model_dimensions)

        # visualize the data using tensor board
        visualizer.visualize()

    @staticmethod
    def get_group_name(index, groupsSizes):
        result = 0
        found = False
        for i, gSize in enumerate(groupsSizes):
            if (index < gSize and found == False):
                found = True
                result = i
        return groupsNames[result]

    @staticmethod
    def read_corpus(corpus, startIndex):
        print 'reading corpus %s' % (corpus)

        i = startIndex

        for filename in glob.glob(os.path.join(corpus, '*.xml')):

            tree = ET.parse(filename)
            root = tree.getroot()

            title = root.find("Title").text
            if title is None:
                title = ""

            body = root.find("Body").text
            if body is None:
                body = ""

            i += 1
            text = title + os.linesep + body
            if i % 100 == 0:
                print 'processed documents: %s' % i

            yield gensim.models.doc2vec.TaggedDocument(
                gensim.utils.lemmatize(text, stopwords=stopwords.words('english')), ntpath.basename(filename))
