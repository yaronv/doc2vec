import os

import numpy as np
import tensorflow as tf
from gensim.models import Doc2Vec
from tensorflow.contrib.tensorboard.plugins import projector

from src.config import *

if __name__ == '__main__':
    print('loading an exiting model')
    model = Doc2Vec.load(doc2vec_model)
    # W = data_helper.load_embedding_vectors_word2vec(model.wv.vocab, word2vec_model, False)

    # adding into projector
    config = projector.ProjectorConfig()
    indexes = []
    print('writing metadata file')
    with open(os.path.join(output_embeddings_graphs_path, 'corpus.tsv'), 'wb') as file_metadata:
        file_metadata.write(b'word\toccurrences' + b'\n')
        for key in sorted(model.wv.vocab, key=lambda word: model.wv.vocab[word].index):
            if model.wv.vocab[key].count >= 7500:
                file_metadata.write((key + '\t' + str(model.wv.vocab[key].count) + '\n').encode('utf-8'))
                indexes.append(model.wv.vocab[key].index)

    placeholder = np.zeros((len(indexes), model_dimensions))
    for i, index in enumerate(indexes):
        placeholder[i] = model.wv.syn0[index]

    print('setting variables for tensorflow')
    embedding_var = tf.Variable(placeholder, trainable=False, name='word-embeddings')
    embed = config.embeddings.add()
    embed.tensor_name = embedding_var.name
    embed.metadata_path = os.path.join(output_embeddings_graphs_path, 'corpus.tsv')

    # define the model without training
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_embeddings_graphs_path, 'w2x_metadata.ckpt'))

    writer = tf.summary.FileWriter(output_embeddings_graphs_path, sess.graph)
    projector.visualize_embeddings(writer, config)

    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_embeddings_graphs_path))