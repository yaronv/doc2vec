from src.doc2vecTR import Doc2VecTR

if __name__ == '__main__':
    unlabeled = '/home/yaron/doc2vec/train-200k'
    # negatives = '/home/yaron/doc2vec/dis-negs'
    dis_positives = '/home/yaron/doc2vec/dis-positives'
    # dis_ignores = '/home/yaron/doc2vec/dis-ignores'
    xpand_positives = '/home/yaron/doc2vec/xpand-positives'
    doc2vec = Doc2VecTR(unlabeled, [dis_positives, xpand_positives])
    doc2vec.run()