# VICTOR: a Dataset for Brazilian Legal Documents Classification

This repo holds source code described in the paper below:

* [Pedro H. Luz de Araujo](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K2742607J6), [Teófilo E. de Campos](https://cic.unb.br/~teodecampos/), [Fabricio Ataides Braz](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4765736Y3), [Nilton Correia da Silva](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4779693P1)
_VICTOR: a Dataset for Brazilian Legal Documents Classification_  
[Language Resources and Evaluation Conference (LREC), May, Marseille, France, 2020.](https://lrec2020.lrec-conf.org/en/)  
Download: [ [paper](https://www.aclweb.org/anthology/2020.lrec-1.181.pdf) | [bib](https://www.aclweb.org/anthology/2020.lrec-1.181.bib) ]

We kindly request that users cite our paper in any publication that is generated as a result of the use of our code or our dataset.

## Requirements
* [Python 3](https://www.python.org/downloads/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
* [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/)
* [keras](https://keras.io/)
* [tensorflow](https://www.tensorflow.org/)

## Files
* shallow_clf_docType.ipynb: notebook to train the shallow classifiers for document type prediction
* baseline_clf_themes.ipynb: notebook to train baseline classifiers for theme prediction
* dataset_statistics.ipynb: notebook to compute dataset statistics
* get_preds.py: script to compute and save model predictions (to use in the CRF experiments)
* crf_experiments.ipynb: notebook for CRF post-processing for document type classification
* train_cnn.py script to train CNN for document type classification
* train_lstm.py script to train LSTM for document type classification
* train_xgboost_themes.py script to train XGBoost for theme classification
