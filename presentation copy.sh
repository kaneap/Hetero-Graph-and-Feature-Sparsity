python graph-learning/experiment_runner.py  --apriori 0 1 --remove_features 0.25 0.5 --model hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py  --apriori 0 1 --remove_features 0.25 0.5 --model gat -e 5000 --deterministic 0 --comment "for presentation"

python graph-learning/experiment_runner.py  --pca 0 1 --remove_features 0.25 0.5 --model hetero_conv -e 5000 --deterministic 0 --comment "for presentation"
python graph-learning/experiment_runner.py  --pca 0 1 --remove_features 0.25 0.5 --model gat -e 5000 --deterministic 0 --comment "for presentation"