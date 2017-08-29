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

        if load_existing:
            print 'loading an exiting model'
            model = Doc2Vec.load(model_path)
        else:
            print 'training a new model'
            corpus_data = list(read_corpus(self.train_corpus, -1))

            model = Doc2Vec(size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)

            model.build_vocab(corpus_data)

            # start training
            for epoch in range(epochs):
                print ('Now training epoch %s' % epoch)
                model.train(corpus_data, total_examples=model.corpus_count, epochs=model.iter)
                # model.alpha -= 0.002  # decrease the learning rate
                # model.min_alpha = model.alpha  # fix the learning rate, no decay

            model.save(model_path)
            model.save_word2vec_format(word2vec_path)

        print 'total docs learned %s' % (len(model.docvecs))

        groups_vectors = []
        ids = []
        labels = []
        groups_sizes = []
        size_so_far = -1

        # add the groups vectors
        for i, group in enumerate(self.groups):
            group_data = list(read_corpus(group, size_so_far))
            size_so_far += len(group_data)
            groups_sizes.append(size_so_far)

            for vec in group_data:
                vec_data = model.infer_vector(vec.words)
                groups_vectors.append(vec_data)
                ids.append(vec.tags)
                labels.append(i)

        print 'writing meta data to file in tensorflow format'
        with open(os.path.join(output_path, meta_file + '.tsv'), 'wb') as file_metadata:
            file_metadata.write('doc_id\tgroup' + '\n')
            for i, id in enumerate(ids):
                file_metadata.write(id + '\t' + self.get_group_name(i, groups_sizes) + '\n')

        print 'writing vectors to file'
        with open(os.path.join(output_path, meta_file + '-vecs.tsv'), 'wb') as file_metadata:
            for i, vec in enumerate(groups_vectors):
                file_metadata.write(",".join(["%.15f" % number for number in vec]) + '\n')

        # print 'writing CNN train data to file'
        # with open(os.path.join(output_path, meta_file + '-train.csv'), 'wb') as file_metadata:
        #     for i, vec in enumerate(groups_vectors):
        #         file_metadata.write(",".join(["%.20f" % number for number in vec]) + ',' + str(labels[i]) + '\n')

        # create a new tensor board visualizer
        visualizer = TF_visualizer(groups_sizes, model_dimensions)

        # visualize the data using tensor board
        visualizer.visualize()

    @staticmethod
    def get_group_name(index, groups_sizes):
        result = 0
        found = False
        for i, g_size in enumerate(groups_sizes):
            if index < g_size and found == False:
                found = True
                result = i
        return groupsNames[result]

def read_corpus(corpus, start_index):
    print 'reading corpus %s' % corpus

    i = start_index

    for filename in glob.glob(os.path.join(corpus, '*.xml')):
        try:
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
        except:
            print 'error parsing file: %s, using naive parsing' % filename
            with open(filename, 'r') as file_content:
                content = file_content.read()
                text = content.replace("<document>", '').replace("</document>", '').replace("<Title>", '').replace("</Title>", '').replace("<Body>", '').replace("</Body>", '')

                yield gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.lemmatize(text, stopwords=stopwords.words('english')), ntpath.basename(filename))