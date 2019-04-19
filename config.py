# -------------------- PATH ---------------------

#ROOT_PATH = "/local/data2/pxu4/TypeClassification"
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
ONTONOTES_DATA_PATH = "%s/OntoNotes" % DATA_PATH
BBN_DATA_PATH="%s/BBN" % DATA_PATH
LOG_DIR = "%s/log" % ROOT_PATH
CHECKPOINT_DIR = "%s/checkpoint" % ROOT_PATH
OUTPUT_DIR = "%s/output" % ROOT_PATH
PKL_DIR='./pkl'

EMBEDDING_DATA = "%s/glove.840B.300d.txt" % DATA_PATH
testemb='testemb'
prep='prep'

# -------------------- DATA ----------------------


ONTONOTES_ALL = "%s/all.txt" % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN = "%s/train.txt" % ONTONOTES_DATA_PATH
ONTONOTES_VALID = "%s/dev.txt" % ONTONOTES_DATA_PATH
ONTONOTES_TEST = "%s/test.txt" % ONTONOTES_DATA_PATH

ONTONOTES_TYPE = "%s/type.pkl" % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN_CLEAN = "%s/train_clean.tsv" % ONTONOTES_DATA_PATH
ONTONOTES_TEST_CLEAN = "%s/test_clean.tsv" % ONTONOTES_DATA_PATH

BBN_ALL = "%s/all.txt" % BBN_DATA_PATH
BBN_TRAIN = "%s/train.txt" % BBN_DATA_PATH
BBN_VALID = "%s/dev.txt" % BBN_DATA_PATH
BBN_TEST = "%s/test.txt" % BBN_DATA_PATH
BBN_TRAIN_CLEAN = "%s/train_clean.tsv" % BBN_DATA_PATH
BBN_TEST_CLEAN = "%s/test_clean.tsv" % BBN_DATA_PATH
BBN_TYPE = "%s/type.pkl" % BBN_DATA_PATH

# --------------------- PARAM -----------------------

MAX_DOCUMENT_LENGTH = 30

MENTION_SIZE = 15

WINDOW_SIZE = 10

RANDOM_SEED = 2017