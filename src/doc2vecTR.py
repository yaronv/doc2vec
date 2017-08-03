import numpy

import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import os
import collections
import smart_open
import random
import glob
import xml.etree.ElementTree as ET
import multiprocessing
import numpy as np
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bhtsne import tsne
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from nltk.corpus import stopwords


class Doc2VecTR(object):

    def __init__(self, train_corpus, groups):
        self.train_corpus = train_corpus
        self.groups = groups


    def run(self):
        print 'app started'

        # root = os.path.abspath(os.path.dirname(__file__))
        # static = os.path.join(root, 'static')


        corpusData = list(self.readCorpus(self.train_corpus, -1))

        cores = multiprocessing.cpu_count()
        print 'num of cores is %s' % (cores)

        model = Doc2Vec(size=400, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)

        model.build_vocab(corpusData)

        # start training
        for epoch in range(20):
            if epoch % 20 == 0:
                print ('Now training epoch %s'%epoch)
            model.train(corpusData, total_examples=model.corpus_count, epochs=model.iter)
            # model.alpha -= 0.002  # decrease the learning rate
            # model.min_alpha = model.alpha  # fix the learning rate, no decay


        model.save('/home/yaron/doc2vec/model/trained.model')
        model.save_word2vec_format('/home/yaron/doc2vec/model/trained.word2vec')

        print 'total docs learned %s' % (len(model.docvecs))

        print 'starting to draw'


        # gather all groups vectors and reduce the dimension
        groupsVectors = []
        groupsSizes = []
        sizeSoFar = -1
        # add the groups vectors
        for group in self.groups:
            groupData = list(self.readCorpus(group, sizeSoFar))
            sizeSoFar += len(groupData)
            groupsSizes.append(sizeSoFar)

            for vec in groupData:
                vecData = model.infer_vector(vec.words)
                groupsVectors.append(vecData)


        # add the corpus vectors
        for vec in model.docvecs:
            groupsVectors.append(vec)


        # X_scaled = preprocessing.scale(groupsVectors)

        pca = sklearnPCA(n_components=2)
        vectors = pca.fit_transform(groupsVectors)



        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # # Make data.
        # X = vectors[:,0]
        # Y = vectors[:,1]
        # X, Y = np.meshgrid(X, Y)
        # Z = vectors[:,2]
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        #
        plt.plot(
            vectors[:,0][:groupsSizes[0]], vectors[:,1][:groupsSizes[0]], 'ro',
            vectors[:,0][groupsSizes[0]:groupsSizes[1]], vectors[:,1][groupsSizes[0]:groupsSizes[1]], 'bo')
            # vectors[:,0][groupsSizes[1]:groupsSizes[2]], vectors[:,1][groupsSizes[1]:groupsSizes[2]], 'go'
        plt.show()

    def readCorpus(self, corpus, startIndex):
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
            if( i % 100 == 0):
                print 'processing document %s' % (i)

            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.lemmatize(text, stopwords=stopwords.words('english')), [i])