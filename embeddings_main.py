from src.config import *
from src.doc2vec_tr import Doc2VecTR

if __name__ == '__main__':

    doc2vec = Doc2VecTR(xpand_all, groups)
    doc2vec.run()