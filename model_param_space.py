
param_space_nfetc_ontonotes_NFETC_CLSC={
    "wpe_dim": 70,
    "hidden_layers": 2,
    "hidden_size": 700,
    "dense_keep_prob": 0.7,
    "rnn_keep_prob": 0.6,
    "num_epochs": 20,
    "makchainlabel": 200,
    'measureway': 'dot-product',
    "lr": 0.0006,
    "state_size": 1000,
    "l2_reg_lambda": 0.000,
    "batch_size": 512,
    "alpha": 0.25,
    'useCCLPloss':True,
    'sslloss':True,
    'hier':True,
    'cclpvar':2.0,
    'makchainfeature':9.0,
    'filterdata':True,
    'bn':False,
}


param_space_nfetc_bbn_NFETC_CLSC={
    "lr": 0.0007,#learning rate
    "state_size": 1000,# LSTM dim
    "l2_reg_lambda": 0.000,# l2 factor
    "alpha": 0.0,# control the hier loss
    'useCCLPloss':True,# use the CLSC or not
    'cclpvar':1.5, # the CLSC factor
    'makchainfeature':13,# the max length of Markov chain
    "wpe_dim": 40,#position embedding dim
    "hidden_layers":1,#number of hidden layer of the classifier
    "hidden_size": 560,#hidden layer dim of the classifier
    "dense_keep_prob": 0.3,#dense dropout rate of the feature extractor
    'rnn_dense_dropout':0.3,#useless
    "rnn_keep_prob": 1.0,# rnn output droput rate
    "batch_size": 512,# as the name
    "num_epochs": 20,# as the name
    "makchainlabel":200,# max time step of label propagation
    'measureway':'dot-product',# the measurement of the distance between samples in the latent space
}

param_space_dict = {
    "nfetc_ontonotes_NFETC_CLSC":param_space_nfetc_ontonotes_NFETC_CLSC,# the best hp for NFETC_CLSC in OntoNotes
    'nfetc_bbn_NFETC_CLSC':param_space_nfetc_bbn_NFETC_CLSC,# the best hp for NFETC_CLSC in BBN
}

int_params = [
    "wpe_dim", "state_size", "batch_size", "num_epochs", "hidden_size", "hidden_layers","sedim","selayer"
]

class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner name!"
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
