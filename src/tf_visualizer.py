import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from src.config import *

class tf_visualizer(object):

    def __init__(self, groupsSizes, dimension):
        self.groupsSizes = groupsSizes
        self.dimension = dimension

    def visualize(self):

        # model = Word2Vec.load(model_path)



        # adding into projector
        config = projector.ProjectorConfig()

        for idx, file in enumerate(metaFiles):
            gSize = self.groupsSizes[idx]
            if(idx > 0):
                gSize = self.groupsSizes[idx] - self.groupsSizes[idx-1]
            placeholder = np.zeros((gSize, self.dimension))

            with open(os.path.join(outputPath, file + '-vecs.tsv'), 'r') as file_metadata:
                for i, line in enumerate(file_metadata):
                    placeholder[i] = np.fromstring(line, sep=',')

            tensor_name = file.replace(".tsv", "")
            embedding_var = tf.Variable(placeholder, trainable = False, name = tensor_name)


            embed = config.embeddings.add()
            embed.tensor_name = embedding_var.name
            embed.metadata_path = os.path.join(outputPath, file + '.tsv')

        # define the model without training
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outputPath, 'w2x_metadata.ckpt'))

        writer = tf.summary.FileWriter(outputPath, sess.graph)
        projector.visualize_embeddings(writer, config)


        print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(outputPath))