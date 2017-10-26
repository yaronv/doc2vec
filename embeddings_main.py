import os

from src.config import *
from src.doc2vec_tr import Doc2VecTR

if __name__ == '__main__':

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    doc2vec = Doc2VecTR(train_500k_random, groups)
    doc2vec.run()
