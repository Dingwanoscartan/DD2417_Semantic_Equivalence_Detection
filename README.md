# DD2417 Semantic Equivalence Detection

This is the mini-project work of the course DD2417 Language Engineering at KTH

Determining whether two questions are asking the same thing can be challenging, as the differences might be very subtle. The choice of words can be quite varying while the semantic meaning stays the same.

we explore different ways to detect semantic equivalence using different GRU models, combined with several DL techniques. We also use a bagging ensemble way to combine our GRU-based models. In the end, we evaluate and compare all these approaches, with our best model achieving 83.1% accuracy on the test set. 

Our dataset is downloaded from Kaggle. The download link: https://www.kaggle.com/datasets/quora/question-pairs-dataset. We use 90% of it as our training set, the rest 10% as our test set.

We use GloVe word embedding. The download link: http://nlp.stanford.edu/data/glove.6B.zip

To train the simGRU model using the best set of hyperparameters that we find:
```
set CUBLAS_WORKSPACE_CONFIG=:4096:8
python SimilarityGRU.py -tr dataset/training.csv -te dataset/test.csv -hs 150 -bs 50 -e 15 -thf 30 -lr 0.001 -ef glove/glove.6B.200d.txt -s -st 1
```
To train the concatGRU model with the best set of parameters coded:
```
python concatGRU.py -tr dataset/training.csv -t dataset/test.csv -wv glove/glove.6B.200d.txt
```
To load the saved models:
```
python SimilarityGRU.py -te dataset/test.csv -l model_folder
```
```
python concatGRU.py -t dataset/test.csv -l model_folder
```
To generate extracted features needed for our Ensemble model:
```
python SimilarityGRU.py -te dataset/test.csv -l model_folder -p destination_folder
```
```
python concatGRU.py -t dataset/test.csv -l model_folder -p destination_folder
```
To run our Ensemble model, you need to firstly change the dir_name in the code to your destination folder.
```
python Ensemble_SED.py
```
