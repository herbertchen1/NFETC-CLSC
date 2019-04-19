from optparse import OptionParser
from task import Task
import logging
from utils import logging_utils
from model_param_space import param_space_dict
import datetime
import config

def parse_args(parser):
    parser.add_option("-m", "--model", dest="model_name", type="string")
    parser.add_option("-d", "--data", dest="data_name", type="string")
    parser.add_option("-p", "--portion", dest="portion", type=int,default=100)
    parser.add_option("-a", "--alpha", dest="alpha", type=float,default=0.)
    parser.add_option("-o", "--savename", dest="save_name", type="string",default='')
    parser.add_option("-r", "--runs", dest="runs", type="int", default=5)
    parser.add_option("-g", "--getfeature", dest="get_features", default=False, action="store_true")#getfeature
    parser.add_option("-i", "--ifretraining", default=False, action="store_true")
    options, args = parser.parse_args()
    return options, args

def main(options):
    time_str = datetime.datetime.now().isoformat()
    if len(options.save_name) == 0:
        logname = "Eval_[Model@%s]_[Data@%s]_%s.log" % (options.model_name,
                    options.data_name, time_str)
    else:
        logname = "Eval_[Model@%s]_[Data@%s]_%s.log" % (options.save_name,
                                                            options.data_name, time_str)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    # else:
    #     time_str = datetime.datetime.now().isoformat()
    #     logname = "Final_[Model@%s]_[Data@%s]_%s.log" % (options.model_name,
    #             options.data_name, time_str)
    #     logger = logging_utils._get_logger(config.LOG_DIR, logname)
    #
    params_dict = param_space_dict[options.model_name]
    params_dict['alpha']=options.alpha
    task = Task(model_name=options.model_name, data_name=options.data_name, cv_runs=options.runs,
                params_dict=params_dict,logger=logger,portion=options.portion,
                save_name=options.save_name)

    print('-'*50+'refit'+'-'*50)
    task.refit()

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
