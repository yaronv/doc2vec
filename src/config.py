# unlabeled documents
train_200k_random = '/home/yaron/doc2vec/train-200k-random'
train_500k_random = '/home/yaron/doc2vec/train-500k-random'
unlabeled = '/home/yaron/doc2vec/unlabeled-200k'
unlabeled_2m = '/home/yaron/doc2vec/2m-data'

# groups of news documents
xpand_positives = '/home/yaron/doc2vec/labeled_data/xpand-positives'
xpand_negs = '/home/yaron/doc2vec/labeled_data/xpand-negs'

awlq_positives= '/home/yaron/doc2vec/labeled_data/awlq-positives'
awlq_negs = '/home/yaron/doc2vec/labeled_data/awlq-negs'

dis_positives= '/home/yaron/doc2vec/labeled_data/dis-positives'
dis_negs = '/home/yaron/doc2vec/labeled_data/dis-negs'
dis_ignores = '/home/yaron/doc2vec/labeled_data/dis-ignores'

dvst_positives= '/home/yaron/doc2vec/labeled_data/dvst-pos'

mrg_positives= '/home/yaron/doc2vec/labeled_data/mrg-pos'

output_path = '/home/yaron/PycharmProjects/doc2vecTF/graphs'
runs_path = '/home/yaron/PycharmProjects/doc2vecTF/runs'
meta_file = 'metadata'

# news documents conf
groupsNames = ['awlq_positives', 'dvst_positives', 'dis_positives', 'xpand_positives', 'mrg_positives']
groups = [awlq_positives, dvst_positives, dis_positives, xpand_positives, mrg_positives]

# groupsNames = ['dis_positives', 'dis_negatives', 'dis_ignores']
# groups = [dis_positives, dis_negs, dis_ignores]

model_path = '/home/yaron/doc2vec/model-200k/model'
doc2vec_model = '{}/doc2vec.model'.format(model_path)
word2vec_model = '{}/word2vec.model'.format(model_path)
word2vec_vocab = '{}/word2vec.vocab'.format(model_path)

model_dimensions = 400

load_existing = True
include_groups_in_train = True
epochs = 50

tokens_separator = "#$#"
