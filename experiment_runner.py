"""
The objective of this code is to provide the minimum configuration setup to run different experiments.
What is a hyperparameter?
    * It is a parameter that you set inside your pipeline and for which the final value is given by you explicitely. 
    * e.g: batch size, embedding size, number of epochs, ... 
What is an experiment?
    * An experiment is the execution of all the steps inside the designed pipeline for a single combination (set) of hyperparameters values. 
    * e.g: we run the pipeline for the combination  batch size = 16, embedding size = 64, number of epochs = 5. Changing at list one value will result in another combination, therefore another experiment. 
How are the hyperparameters set for one experiment?
    * From the terminal: python3 name_of_the_file.py -short_param1 value1_param1 value2_param1 value3_param1 --long_param2 value1_param2 value2_param2  
    * In debug mode with VSCode:
        - Run and Debug (Ctrl+Shift+D)
        - create a launch.json file (generally it goes inside root/.vscode)
        - Structure of the launch.json file
            {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Current File",
                        "type": "debugpy",
                        "request": "launch",
                        "program": "${file}",
                        "console": "integratedTerminal",
                        "justMyCode": true
                    }
            }
            The 'Python: Current File' configuration is important to be able to always run and debug the current file
        - Add a configuration for a specific file: appending it to the dictionary
            {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Current File",
                        "type": "debugpy",
                        "request": "launch",
                        "program": "${file}",
                        "console": "integratedTerminal",
                        "justMyCode": true
                    },
                    {
                        "name": "<Name of the Configuration>",  <------ change
                        "type": "debugpy",
                        "request": "launch",
                        "program": "<relative_path_of_the_python_file.py>",     <------ change
                        "console": "integratedTerminal",
                        "args": [
                            "--long_param1",
                            "value1_param1", "value2_param1",
                            "-short_param2",
                            "value1_param2", "value2_param2", "value3_param2",
                        ],
                        "justMyCode": false
                    },
            }
How many experiments are going to be run?
    * The total number of experiments is given by the product of the number of values inserted for each parameter.
    * e.g: --batch_size 16 32 --embedding_size 64 128 256, --number_of_epochs 5 --> Total = 3*2*1 = 6 experiments
    * Each parameter represents a set of values and by combining them, we get the cartesian productof all sets.
"""

# %%
import os
import configargparse
import copy

import torch
import trainer
from utils.set_seed import set_seed
from utils import get_dataset
from utils.get_model import get_model
from enhancements.new_edges import add_new_edges
from enhancements.compress_add import compress_add_feats
from enhancements.feature_prop_2 import propagate_features, propagate_features_homogeneous
from enhancements.orphan_fixer import add_edge_to_orphans
from enhancements.similar_nodes_connector import connect_similar_nodes
from link_pred_enhancement import predict_links
from utils import graph_polluters
from enhancements import apriori
from modules.pca import PCA

from datetime import datetime
import pandas as pd

savepath = 'results/results.csv'

# %%
# Here is where we dedine the name, type and default values of our hyperparameters. All the hyperparameters shouldbe defined 
# in order to be able to use them. short means the short version of the name, long is the long version of the name.
def setup_config(config):
    print('Configuration setup ...')
    config._parser.add("-d", "--dataset_name", default='dblp', type=str, help="The name of the dataset to train on", nargs='*')
    config._parser.add("-m", "--model", default='gat', type=str, help="gat, hetero_conv, hgt", nargs='*')
    config._parser.add("-b", "--batch_size", default=64, type=int, help="The size of the batch during the training process", nargs='*')
    config._parser.add("-e", "--number_of_epochs", default=10, type=int, help="The number of epochs to train the model", nargs='*')
    config._parser.add("-lr", "--learning_rate", default=0.005, type=float, help="The learning rate to train the model", nargs='*')
    config._parser.add("-wd", "--weight_decay", default=0.001, type=float, help="The weight decay for the optimizer", nargs='*')
    config._parser.add("-rf", "--remove_features", default=0.0, type=float, help="The ratio of features to set to zero", nargs='*')
    config._parser.add("-re", "--remove_edges", default=0.0, type=float, help="The ratio of edges to set to zero", nargs='*')
    config._parser.add("-eg", "--graph_enhancement", default=0, type=int, help="The graph enhancement method", nargs='*')
    config._parser.add("-ef", "--feature_enhancement", default=0, type=int, help="The node enhancement method", nargs='*')
    config._parser.add("-lp", "--link_prediction", default=0, type=int, help="Whether to apply link prediction", nargs='*')
    config._parser.add("-pe", "--predefined_enhancement", default='none', type=str, help="enhance_g enhance_f none", nargs='*')
    config._parser.add("-s", "--seed", default=42, type=int, help="The random seed to set", nargs='*')
    config._parser.add("-det", "--deterministic", default=1, type=int, help="Whether to use deterministic algorithms")
    config._parser.add("-c", "--comment", default='', type=str, help="A comment to add to the rows of the results")
    config._parser.add("-ap", "--apriori", default=0, type=int, help="whether to use apriori enhancement", nargs='*')
    config._parser.add("-pca", "--pca", default=0, type=int, help="whether to use pca enhancement", nargs='*')
    config._parser.add("-pf", "--propagate_features", default=0, type=int, help="whether to use feature propagation", nargs='*')
    config._parser.add("-ce", "--continuous_enhancement", default=0, type=int, help="number of iterations to do continuous enhancement", nargs='*')
    config._parser.add("-fo", "--fix_orphans", default=0, type=int, help="whether to fix orphan nodes", nargs='*')

    config.parse()

# Here we create the mechanism to generate all combinations of parameters, one for each experiment
class Configuration:
    def __init__(self):
        self._parser = configargparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []       
        
    def parse(self):
        self._config = self._parser.parse_args()
    
        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    connectionrent = copy.deepcopy(c)
                    setattr(connectionrent, ma, v)
                    new_configs.append(connectionrent)

            # store splitted values
            self._configs = new_configs
        
    def get_configs(self):
        return self._configs

def augment_data(data, args):

    return data

def save_result(file_path, results, columns):
    if os.path.exists(file_path):
        current_df = pd.read_csv(file_path)
    else:
        current_df = pd.DataFrame(columns=columns)

    filtered_results = {key: results[key] for key in columns if key in results}
    filtered_results['date_time'] = datetime.now()
    print(filtered_results)
    new_df = pd.DataFrame(data=filtered_results, index=[0])
    final_df = pd.concat([current_df,new_df], axis=0)
    final_df.to_csv(file_path, index=False)


# Here we implement the entire code of our pipeline
def run_pipeline(args, experiment_name):
    
    exp_starttime = datetime.now()
    data = get_dataset.get_dataset(args.dataset_name)

    target_node_type = get_dataset.get_target_node_type(args.dataset_name)

    # pollute the data
    node_types_to_remove = get_dataset.get_node_types_to_remove(args.dataset_name)
    data, masks = graph_polluters.remove_features(data, args.remove_features, node_types_to_remove)
    # masks represents the edges with features removed
    if args.remove_edges > 0:
        edge_type_to_remove, rev_edge_type_to_remove = get_dataset.get_edge_types_to_remove(args.dataset_name)
        data = graph_polluters.remove_edges(data, args.remove_edges, edge_type_to_remove, rev_edge_type_to_remove)


    edge_type_to_remove, lp_rev_edge_type = get_dataset.get_lp_edge_types(args.dataset_name)

    if args.fix_orphans:
        add_edge_to_orphans(data, ('author','to','paper'), ('paper','to','author'))

    if args.propagate_features:
            propagate_features(data, masks, 'author', ('author','to','paper'), ('paper','to','author'), iters=40)
            propagate_features(data, masks, 'paper', ('paper','to','author'), ('author','to','paper'), iters=40)
            propagate_features(data, masks, 'term', ('term','to','paper'), ('paper','to','term'), iters=40)
    '''
    for i in range(args.continuous_enhancement):
        connect_similar_nodes(data, 'author', ('author','sim','author'),0.3)
        print(data)
        propagate_features_homogeneous(data, masks, 'author', ('author','sim','author'), iters=40)
    '''


    if args.graph_enhancement:
        new_edge_types = add_new_edges(data, device, target_node_type, num_paths=1)
        
        for edge_type in new_edge_types:
            node_type,_,_ = edge_type
            propagate_features_homogeneous(data, masks, node_type, edge_type, iters=40, device=device)
        for i in range (40):
            connect_similar_nodes(data, 'author', ('author','sim','author'),0.3)
            propagate_features_homogeneous(data, masks, 'author', ('author','sim','author'), iters=1, device=device)
    if args.link_prediction == 1:
        data = predict_links(data, device, threshold=0.9, edge_type=edge_type_to_remove, rev_edge_type=lp_rev_edge_type)
    if args.feature_enhancement:
        node_types_to_enhance = get_dataset.get_node_types_to_enhance(args.dataset_name)
        data = compress_add_feats(data, node_types_to_enhance, method=args.feature_enhancement)
    if args.link_prediction == 2:
        data = predict_links(data, device, threshold=0.9,  edge_type=edge_type_to_remove, rev_edge_type=lp_rev_edge_type)
    if args.apriori:
        item_sets = apriori.find_frequent_sets(data, 'author', min_support = 0.01)
        apriori.conflate_item_sets(data, 'author', item_sets)
    if args.pca:
        data['author'].x = PCA(18).to(device).fit_transform(data['author'].x)
        data['term'].x = PCA(18).to(device).fit_transform(data['term'].x)
        data['paper'].x = PCA(18).to(device).fit_transform(data['paper'].x)

    


    model, opimizer = get_model(args.model, data, device, target_node_type, args.learning_rate, args.weight_decay)

    results = trainer.train(data, device, model, opimizer, args.number_of_epochs, experiment_name, target_node_type)
    #results = trainer.train_2(data, device, args, experiment_name, target_node_type)

    results = results | args.__dict__
    results['args'] = str(args)
    results['experiment_duration'] = str(exp_starttime - datetime.now())
    results['experiment_name'] = experiment_name
    save_result('graph-learning/results/results.csv', results, schema_columns)
    return

#datetime.now()
schema_columns = ['test_prec', 'test_recall', 'test_f1', 'test_acc',
                   'test_loss', 'train_acc', 'train_f1', 'train_loss',
                   'val_acc', 'val_f1', 'val_loss', 'dataset_name', 
                   'model', 'batch_size', 'number_of_epochs', 'best_epoch',
                   'experiment_duration', 'learning_rate', 'weight_decay', 'remove_features', 
                   'remove_edges', 'graph_enhancement', 'feature_enhancement', 'link_prediction',
                   'continucus_enhancement',
                   'predefined_enhancement', 'experiment_name', 'date_time', 'comment', 'args']

# %%
# Here we start the evaluation
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Configuration() 
    setup_config(config)
    evaluation_starttime = str(datetime.now())
    exp_num = 1
    tot_exp = len(config.get_configs())
    for args in config.get_configs():
        print(f'Starting experiment number {exp_num}/{tot_exp} ...')
        if args.deterministic:
            set_seed(args.seed)
        enhancement_type = {'enhance_g': [1,0],'enhance_n': [0,1],'none':[args.graph_enhancement, args.feature_enhancement]}
        args.graph_enhancement, args.feature_enhancement = enhancement_type[args.predefined_enhancement]
        print(args)
        experiment_name = evaluation_starttime + f' exp {exp_num:03d}'
        run_pipeline(args, experiment_name=experiment_name)
        
        exp_num += 1    # increment the counter
    print('Completed')