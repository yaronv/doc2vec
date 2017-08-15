# unlabeled documents
unlabeled = '/home/yaron/doc2vec/train-200k'

# groups of news documents
dis_negatives = '/home/yaron/doc2vec/docs-topics/dis-negs2'
dis_positives = '/home/yaron/doc2vec/docs-topics/dis-positives'
dis_ignores = '/home/yaron/doc2vec/docs-topics/dis-ignores'
xpand_positives = '/home/yaron/doc2vec/docs-topics/xpand-positives'
xpand_ignores = '/home/yaron/doc2vec/docs-topics/xpand-ignores'
xpand_negs = '/home/yaron/doc2vec/docs-topics/xpand-negs'
awlq_positives= '/home/yaron/doc2vec/docs-topics/awlq-positives'
awlq_ignores = '/home/yaron/doc2vec/docs-topics/awlq-ignores'
awlq_negs = '/home/yaron/doc2vec/docs-topics/awlq-negs'
mrg_positives= '/home/yaron/doc2vec/docs-topics/mrg-pos'
dvst_positives= '/home/yaron/doc2vec/docs-topics/dvst-pos'
brib_pos = '/home/yaron/doc2vec/docs-topics/brib-pos'
class_pos = '/home/yaron/doc2vec/docs-topics/class-pos'
fake1_pos = '/home/yaron/doc2vec/docs-topics/fake1-pos'

# groups of rr sentences
acci_sents_neg = '/yaron/doc2vec/sentences-topics/acci/neg_sents'
acci_sents_pos = '/yaron/doc2vec/sentences-topics/acci/pos_sents'
dvst_sents_neg = '/home/yaron/doc2vec/sentences-topics/dvst/neg_sents'
dvst_sents_pos = '/home/yaron/doc2vec/sentences-topics/dvst/pos_sents'
layofs_sents_neg = '/home/yaron/doc2vec/sentences-topics/layofs/layofs_neg_sents'
layofs_sents_pos = '/home/yaron/doc2vec/sentences-topics/layofs/layofs_pos_sents(tagged)'
mrg_sents_neg = '/home/yaron/doc2vec/sentences-topics/mrg/neg_sents'
mrg_sents_pos = '/home/yaron/doc2vec/sentences-topics/mrg/pos_sents'

output_path = '/home/yaron/PycharmProjects/doc2vecTF/graphs'
meta_file = 'metadata'

# documents conf
# groupsNames = ['dis_positives', 'xpand_positives', 'awlq_positives', 'mrg_positives', 'dvst_positives', 'brib_positives', 'class_positives', 'fake1_positives']
# groups = [dis_positives, xpand_positives, awlq_positives, mrg_positives, dvst_positives, brib_pos, class_pos, fake1_pos]

# sentences conf
groupsNames = ['acci_pos', 'dvst_pos', 'layofs_pos', 'mrg_pos']
groups = [acci_sents_pos, dvst_sents_pos, layofs_sents_pos, mrg_sents_pos]

model_path = '/home/yaron/doc2vec/model/trained.model'
word2vec_path = '/home/yaron/doc2vec/model/trained.word2vec'
model_dimensions = 400

loadExisting = True
epochs = 50