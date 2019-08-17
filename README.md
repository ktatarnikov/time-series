# TimeSeries tools

The tiny collection of algorithms to work with time series.

- Preprocessing and Feature Engineering tools
-- tsfresh https://tsfresh.readthedocs.io/en/latest/
-- imbalance learn https://imbalanced-learn.readthedocs.io/en/stable/

- Gradient Boosting tree model
  https://xgboost.readthedocs.io/en/latest/

- LSTM Autoencoder
  https://blog.keras.io/building-autoencoders-in-keras.html

- SAX-PAA and discords
  Pavel Senin et al. "Time series anomaly discovery with grammar-based compression." In: EDBT. 2015, pp. 481–492
  (https://openproceedings.org/2015/conf/edbt/paper-155.pdf)

- SSH and minhash
  [NIPS Time Series Workshop 2016] SSH (Sketch, Shingle, & Hash) for Indexing Massive-Scale Time Series.
    by Chen Luo, Anshumali Shrivastava (https://arxiv.org/abs/1610.07328)

- Wighted minhash
  "Improved consistent sampling, weighted minhash and l1 sketching.", by Ioffe, Sergey.
    Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.  

# Installation

- git clone ...
- pip install -r requirements.txt
 - run `jupyter notebook`
 
# Notebooks

 - Anomaly Classification (notebooks/Classification of Anomalous TimeSeries.ipynb)
 - TimeSeries Similarity (notebooks/TimeSeries similarity.ipynb)
 - SAX Anomaly (notebooks/Anomaly-Sequitur.ipynb)
