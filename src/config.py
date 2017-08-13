# unlabeled documents
unlabeled = '/home/yaron/doc2vec/train-200k'

# groups of documents
dis_negatives = '/home/yaron/doc2vec/dis-negs2'
dis_positives = '/home/yaron/doc2vec/dis-positives'
dis_ignores = '/home/yaron/doc2vec/dis-ignores'
xpand_positives = '/home/yaron/doc2vec/xpand-positives'
xpand_ignores = '/home/yaron/doc2vec/xpand-ignores'
xpand_negs = '/home/yaron/doc2vec/xpand-negs'

awlq_positives= '/home/yaron/doc2vec/awlq-positives'
awlq_ignores = '/home/yaron/doc2vec/awlq-ignores'
awlq_negs = '/home/yaron/doc2vec/awlq-negs'

outputPath = '/home/yaron/PycharmProjects/doc2vecTF/graphs'
metaFiles = ['dis_pos_metadata', 'dis_ign_metadata']


groups = [dis_positives, dis_ignores]
colors = ['bo', 'ro', 'go']
markers = ['o', '^', 'o']
marker_size = 5

model_path = '/home/yaron/doc2vec/model/trained.model'
word2vec_path = '/home/yaron/doc2vec/model/trained.word2vec'

loadExisting = True
epoches = 50