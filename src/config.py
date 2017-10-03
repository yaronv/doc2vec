# unlabeled documents
unlabeled = '/home/yaron/doc2vec/train-200k'
unlabeled_2m = '/home/yaron/doc2vec/2m-data'

# groups of news documents
dis_negatives = '/home/yaron/doc2vec/docs-topics/dis-negs2'
dis_positives = '/home/yaron/doc2vec/docs-topics/dis-positives'
dis_ignores = '/home/yaron/doc2vec/docs-topics/dis-ignores'
xpand_positives = '/home/yaron/doc2vec/docs-topics/xpand-positives'
xpand_ignores = '/home/yaron/doc2vec/docs-topics/xpand-ignores'
xpand_negs = '/home/yaron/doc2vec/docs-topics/xpand-negs'
xpand_all = '/home/yaron/doc2vec/docs-topics/xpand-all'
awlq_positives= '/home/yaron/temp/awlq-pos/xml'
awlq_ignores = '/home/yaron/doc2vec/docs-topics/awlq-ignores'
awlq_negs = '/home/yaron/temp/awlq-negs/xml'
mrg_positives= '/home/yaron/doc2vec/docs-topics/mrg-pos'
dvst_positives= '/home/yaron/doc2vec/docs-topics/dvst-pos'
brib_pos = '/home/yaron/doc2vec/docs-topics/brib-pos'
class_pos = '/home/yaron/doc2vec/docs-topics/class-pos'
fake1_pos = '/home/yaron/doc2vec/docs-topics/fake1-pos'

# groups of sentences
acci_sents_neg = '/home/yaron/doc2vec/sentences-topics/acci/neg_sents'
acci_sents_pos = '/home/yaron/doc2vec/sentences-topics/acci/pos_sents'
dvst_sents_neg = '/home/yaron/doc2vec/sentences-topics/dvst/neg_sents'
dvst_sents_pos = '/home/yaron/doc2vec/sentences-topics/dvst/pos_sents'
layofs_sents_neg = '/home/yaron/doc2vec/sentences-topics/layofs/layofs_neg_sents'
layofs_sents_pos = '/home/yaron/doc2vec/sentences-topics/layofs/layofs_pos_sents(tagged)'
mrg_sents_neg = '/home/yaron/doc2vec/sentences-topics/mrg/neg_sents'
mrg_sents_pos = '/home/yaron/doc2vec/sentences-topics/mrg/pos_sents'

# groups of research reports docs
dvst_rr_docs_pos = '/home/yaron/doc2vec/research-reports-docs/dvst/pos_docs'
dvst_rr_docs_neg = '/home/yaron/doc2vec/research-reports-docs/dvst/neg_docs'

output_path = '/home/yaron/PycharmProjects/doc2vecTF/graphs'
runs_path = '/home/yaron/PycharmProjects/doc2vecTF/runs'
meta_file = 'metadata'

# news documents conf
# groupsNames = ['dis_positives', 'xpand_positives', 'awlq_positives', 'mrg_positives', 'dvst_positives', 'brib_positives', 'class_positives', 'fake1_positives']
# groups = [dis_positives, xpand_positives, awlq_positives, mrg_positives, dvst_positives, brib_pos, class_pos, fake1_pos]

groupsNames = ['awlq_positives', 'awlq_negatives']
groups = [awlq_positives, awlq_negs]

# research reports documents conf
# groupsNames = ['dvst_rr_docs_pos', 'dvst_rr_docs_neg']
# groups = [dvst_rr_docs_pos, dvst_rr_docs_neg]

# sentences conf
# groupsNames = ['acci_pos', 'dvst_pos', 'layofs_pos', 'mrg_pos']
# groups = [acci_sents_pos, dvst_sents_pos, layofs_sents_pos, mrg_sents_pos]

model_path = '/home/yaron/doc2vec/model-2m/trained.model'
word2vec_path = '/home/yaron/doc2vec/model-2m/trained.word2vec'

word2vec_model = '/home/yaron/doc2vec/model-2m/word2vec.model'
word2vec_vocab = '/home/yaron/doc2vec/model-2m/word2vec.vocab'

model_dimensions = 400

load_existing = False
epochs = 10

tokens_separator = "#$#"