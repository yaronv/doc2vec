import datetime
import os
import time

import tensorflow as tf
import numpy as np
from gensim.models import Doc2Vec, Word2Vec
from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary
from tensorflow.contrib.tensorboard.plugins import projector

from src import data_helper
from src.config import *
from src.text_cnn import TextCNN
from tensorflow.contrib import learn
import gensim

if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Data loading params
    tf.flags.DEFINE_float("test_sample_percentage", .1, "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string("positive_data_folder", xpand_positives, "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_folder", xpand_negs, "Data source for the negative data.")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", model_dimensions, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print ''

    # Data Preparation
    # ==================================================
    # Load the doc2vec and word2vec models
    print 'Loading words model...'
    # model = Doc2Vec.load(model_path)
    word_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    print ''

    # Load data
    print 'Loading data...'
    x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_folder, FLAGS.negative_data_folder)
    print ''

    # Build vocabulary
    # sorted_model_vocab = sorted(word_model.vocab.items(), key=lambda item:item[1].index)
    # vocab = CategoricalVocabulary()
    # [vocab.get(item[0]) for item in sorted_model_vocab]
    # vocab.freeze()

    max_document_length = max([len(x.split(tokens_separator)) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length, tokenizer_fn=data_helper.my_tokenizer)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Get the existing word embeddings into a matrix
    vocab_len = len(vocab_processor.vocabulary_)
    embedding_matrix = np.zeros((vocab_len, FLAGS.embedding_dim))

    # Output directory for models and summaries
    timestamp = str(int(time.time()))

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))

    not_found = 0
    with open(os.path.join(out_dir, meta_file + '.tsv'), 'w') as metadata_file:
        for i in range(vocab_len):
            word = vocab_processor.vocabulary_.reverse(i)
            metadata_file.write('%s\n' % word)
            if word_model.__contains__(word):
                embed = word_model[word]
                embedding_matrix[i] = embed
            else:
                not_found += 1
                # print 'embedding not found for word: %s' %word
    print 'Total words not found in embeddings: {:d}'.format(not_found)

    # projecting the embeddings
    config = projector.ProjectorConfig()
    embedding_var = tf.Variable(embedding_matrix, trainable=False, name="embeddings")
    embed = config.embeddings.add()
    embed.tensor_name = embedding_var.name
    embed.metadata_path = os.path.join(out_dir, meta_file + '.tsv')
    projector.visualize_embeddings(tf.summary.FileWriter(out_dir), config)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(len(y))
    x_shuffled = np.array(x)[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_len,
                embeddings=embedding_matrix,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "test")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_test, y_test, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
