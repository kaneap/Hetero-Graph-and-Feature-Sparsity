python graph-learning/experiment_runner.py --dataset_name imdb --feature_enhancement 0 --graph_enhancement 0 --remove_features 0.25 --remove_edges 0.25 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py --dataset_name imdb --feature_enhancement 0 --graph_enhancement 0 --remove_features 0.5 --remove_edges 0.5 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"

python graph-learning/experiment_runner.py --dataset_name imdb --feature_enhancement 3 --graph_enhancement 1 --remove_features 0.25 --remove_edges 0.25 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py --dataset_name imdb --feature_enhancement 3 --graph_enhancement 1 --remove_features 0.5 --remove_edges 0.5 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"

python graph-learning/experiment_runner.py --dataset_name imdb --apriori 3 --remove_features 0.25 --remove_edges 0.25 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py --dataset_name imdb --apriori 3 --remove_features 0.5 --remove_edges 0.5 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"

python graph-learning/experiment_runner.py --dataset_name imdb --pca 3 --remove_features 0.25 --remove_edges 0.25 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py --dataset_name imdb --pca 3 --remove_features 0.5 --remove_edges 0.5 --model gat hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
