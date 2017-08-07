import glob
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import multiprocessing
import os
import xml.etree.ElementTree as ET
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA

import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords

from src.config import *


class Doc2VecTR(object):

    def __init__(self, train_corpus, groups):
        self.train_corpus = train_corpus
        self.groups = groups


    def run(self):
        print 'app started'

        # root = os.path.abspath(os.path.dirname(__file__))
        # static = os.path.join(root, 'static')


        cores = multiprocessing.cpu_count()
        print 'num of cores is %s' % (cores)



        if(loadExisting):
            print 'loading an exiting model'
            model = Doc2Vec.load(model_path)
            # word_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
        else:
            print 'training a new model'
            corpusData = list(self.readCorpus(self.train_corpus, -1))

            model = Doc2Vec(size=400, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)

            model.build_vocab(corpusData)

            # start training
            for epoch in range(epoches):

                print ('Now training epoch %s'%epoch)
                model.train(corpusData, total_examples=model.corpus_count, epochs=model.iter)
                # model.alpha -= 0.002  # decrease the learning rate
                # model.min_alpha = model.alpha  # fix the learning rate, no decay


            model.save(model_path)
            model.save_word2vec_format(word2vec_path)

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

        pca = sklearnPCA(n_components=3)
        vectors = pca.fit_transform(groupsVectors)



        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        for idx, group in enumerate(groups):

            fromIndex = 0
            if idx > 0:
                fromIndex = groupsSizes[idx-1]

            # ax.scatter(vectors[:, 0][fromIndex:groupsSizes[idx]], vectors[:, 1][fromIndex:groupsSizes[idx]], vectors[:, 2][fromIndex:groupsSizes[idx]],
            #              c=colors[idx], marker=markers[idx])

            plt.plot(vectors[:,0][fromIndex:groupsSizes[idx]], vectors[:,1][fromIndex:groupsSizes[idx]], colors[idx], markersize=marker_size)
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