import numpy as np
import re
import itertools
from collections import Counter

from src.config import tokens_separator
from src.doc2vec_tr import read_corpus


def getLabelArrayFromGroup(lbl):
    if lbl == 0:
        return [1,0]
    elif lbl == 1:
        return [0,1]
    else:
        return [0,0]


def load_data_and_labels(group_0_folder, group_1_folder):
    group_0_data = list(read_corpus(group_0_folder, -1))
    group_1_data = list(read_corpus(group_1_folder, len(group_0_data)))

    train_data = []
    labels = []

    for doc in group_0_data:
        train_data.append(tokens_separator.join(map(str, doc.words)))
        labels.append(getLabelArrayFromGroup(0))

    for doc in group_1_data:
        train_data.append(tokens_separator.join(map(str, doc.words)))
        labels.append(getLabelArrayFromGroup(1))

    return [train_data, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def my_tokenizer(iterator):
    """Tokenizer generator.
    Args:
      iterator: Input iterator with strings.
    Yields:
      array of tokens per each value in the input.
    """
    for value in iterator:
        yield value.split(tokens_separator)