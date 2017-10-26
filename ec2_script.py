import gc
import itertools
import multiprocessing
import ntpath
import os
import xml.etree.ElementTree as ET
from random import shuffle

import gensim
import numpy as np
import tensorflow as tf
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from tensorflow.contrib.tensorboard.plugins import projector

print ('setting configurations')

train_500k_random = '/home/ubuntu/DL/Data/train-500k-random'

# groups of news documents
xpand_positives  = '/home/ubuntu/DL/Data/labeled_data/xpand-pos/xml'
xpand_negs       = '/home/ubuntu/DL/Data/labeled_data/xpand-negs/xml'

awlq_positives   = '/home/ubuntu/DL/Data/labeled_data/awlq-pos/xml'
awlq_negs        = '/home/ubuntu/DL/Data/labeled_data/awlq-negs/xml'

blktrd_positives = '/home/ubuntu/DL/Data/labeled_data/blktrd-pos/xml'
blktrd_negs      = '/home/ubuntu/DL/Data/labeled_data/blktrd-negs/xml'

trd_positives    = '/home/ubuntu/DL/Data/labeled_data/trd-pos/xml'
trd_negs         = '/home/ubuntu/DL/Data/labeled_data/trd-negs/xml'

output_path_base = '/home/ubuntu/DL/outputs'
output_path = '{}/graphs'.format(output_path_base)
runs_path   = '{}/runs'.format(output_path_base)
meta_file   = 'metadata'

# news documents conf
groupsNames = ['awlq_positives', 'blktrd_positives', 'trd_positives', 'xpand_positives']
groups = [awlq_positives, blktrd_positives, trd_positives, xpand_positives]

model_path = '/home/ubuntu/DL/outputs/model-500k'
doc2vec_model = '{}/doc2vec.model'.format(model_path)
word2vec_model = '{}/word2vec.model'.format(model_path)

model_dimensions = 400

load_existing = False
include_groups_in_train = True
epochs = 80


class DocumentsIterable(object):
    def __init__(self, dirnames):
        self.dirnames = dirnames
        self.files = [os.listdir(dirname) for dirname in dirnames]

        # make sure the files contain their full path
        for i, files in enumerate(self.files):
            self.files[i] = [os.path.join(self.dirnames[i], f) for f in self.files[i]]

        self.files = list(itertools.chain(*self.files))

    def __iter__(self):
        i = 0
        print('starting to process docs (iterating)')
        shuffle(self.files)
        for filename in self.files:
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
                if i % 5000 == 0:
                    print('processed documents: %s' % i)
                words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                yield gensim.models.doc2vec.TaggedDocument(
                    [w.decode('utf-8') for w in words], tags=ntpath.basename(filename))
            except:
                print('error parsing file: %s, using naive parsing' % filename)
                with open(filename, 'r') as file_content:
                    content = file_content.read()
                    text = content.replace("<document>", '').replace("</document>", '').replace("<Title>",
                                                                                                '').replace(
                        "</Title>", '').replace("<Body>", '').replace("</Body>", '')
                    words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                    yield gensim.models.doc2vec.TaggedDocument(
                        [w.decode('utf-8') for w in words], tags=ntpath.basename(filename))


class CorpusReader(object):
    def __init__(self, dirnames):
        self.docs = []
        self.dirnames = dirnames
        self.files = [os.listdir(dirname) for dirname in dirnames]

        # make sure the files contain their full path
        for i, files in enumerate(self.files):
            self.files[i] = [os.path.join(self.dirnames[i], f) for f in self.files[i]]

        self.files = list(itertools.chain(*self.files))

    def read_corpus(self):
        i = 0
        print('starting to process docs (iterating)')
        for filename in self.files:
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
                    print('processed documents: %s' % i)
                words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                self.docs.append(gensim.models.doc2vec.TaggedDocument(
                    [w.decode('utf-8') for w in words], tags=[ntpath.basename(filename)]))
            except:
                print('error parsing file: %s, using naive parsing' % filename)
                with open(filename, 'r') as file_content:
                    content = file_content.read()
                    text = content.replace("<document>", '').replace("</document>", '').replace("<Title>",
                                                                                                '').replace(
                        "</Title>", '').replace("<Body>", '').replace("</Body>", '')
                    words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                    self.docs.append(gensim.models.doc2vec.TaggedDocument(
                        [w.decode('utf-8') for w in words], tags=[ntpath.basename(filename)]))
        return self.docs

class TF_visualizer(object):
    def __init__(self, groups_sizes, dimension):
        self.groups_sizes = groups_sizes
        self.dimension = dimension

    def visualize(self):

        # adding into projector
        config = projector.ProjectorConfig()

        g_size = list(self.groups_sizes.values())[-1]

        placeholder = np.zeros((g_size, self.dimension))

        with open(os.path.join(output_path, meta_file + '-vecs.tsv'), 'r') as file_metadata:
            for i, line in enumerate(file_metadata):
                if(line != ''):
                    placeholder[i] = np.fromstring(line, sep=',')

        tensor_name = meta_file
        embedding_var = tf.Variable(placeholder, trainable=False, name=tensor_name)

        embed = config.embeddings.add()
        embed.tensor_name = embedding_var.name
        embed.metadata_path = os.path.join(output_path, meta_file + '.tsv')

        # define the model without training
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))

        writer = tf.summary.FileWriter(output_path, sess.graph)
        projector.visualize_embeddings(writer, config)

        print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

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
            model = Doc2Vec(size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores,
                            dm=1)
            print('building vocabulary...')
            model.build_vocab(corpus_data)

            # start training the model
            for epoch in range(epochs):
                print('Now training epoch %s' % epoch)
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
                groups_sizes[i] = groups_sizes[i - 1]

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

# Training the model
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

doc2vec = Doc2VecTR(train_500k_random, groups)
doc2vec.run()
