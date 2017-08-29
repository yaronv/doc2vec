import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from src.config import *


class TF_visualizer(object):
    def __init__(self, groups_sizes, dimension):
        self.groups_sizes = groups_sizes
        self.dimension = dimension

    def visualize(self):

        # adding into projector
        config = projector.ProjectorConfig()

        g_size = self.groups_sizes[len(self.groups_sizes) - 1] + 1

        placeholder = np.zeros((g_size, self.dimension))

        with open(os.path.join(output_path, meta_file + '-vecs.tsv'), 'r') as file_metadata:
            for i, line in enumerate(file_metadata):
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
