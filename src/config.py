# unlabeled documents
unlabeled = '/home/yaron/doc2vec/train-200k'

# groups of documents
dis_negatives = '/home/yaron/doc2vec/dis-negs2'
dis_positives = '/home/yaron/doc2vec/dis-positives'
dis_ignores = '/home/yaron/doc2vec/dis-ignores'
xpand_positives = '/home/yaron/doc2vec/xpand-positives'

groups = [dis_negatives, dis_positives, dis_ignores]
colors = ['r', 'b', 'g']
markers = ['o', 'o', 'o']
labels = ['positives', 'ignores']
marker_size = 5

model_path = '/home/yaron/doc2vec/model/trained.model'
word2vec_path = '/home/yaron/doc2vec/model/trained.word2vec'

loadExisting = True
epoches = 50