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


output_path = '/home/yaron/PycharmProjects/doc2vecTF/graphs'
meta_file = 'metadata'

groupsNames = ['dis_positives', 'dis_ignores', 'xpand_positives', 'xpand_ignores']
groups = [dis_positives, dis_ignores, xpand_positives, xpand_ignores]

model_path = '/home/yaron/doc2vec/model/trained.model'
word2vec_path = '/home/yaron/doc2vec/model/trained.word2vec'
model_dimensions = 400

loadExisting = True
epoches = 50