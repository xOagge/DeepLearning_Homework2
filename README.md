# Deep Learning Homework 2 — RNA Binding Affinity Prediction

Predicting RNA-protein binding affinity for RBFOX1 using sequence-based deep learning models.

## Models implemented
- **CNN**: 3 convolutional layers with batch normalization, global max+average pooling
- **BiLSTM**: 2-layer bidirectional LSTM with batch normalization and dropout
- **Attention-LSTM**: BiLSTM extended with custom multi-head additive attention mechanism

## Best results
Attention-LSTM (2 heads, lr=5e-4, hidden=256, dropout=0.3):
- Val Spearman: 0.6804
- Test Spearman: 0.6794
- Test MSE: 0.3136
- Best epoch: 48/50

## Dataset
RNAcompete — ~241k RNA 41-mers for RBFOX1 protein. Split: 96,261 train / 24,065 val / 121,031 test.

## Environment (giljorge0)
Kaggle GPU notebook — Tesla P100-PCIE-16GB, Python 3.12, PyTorch

## Key files
- `homework2-q2.ipynb` — main notebook: model definitions, training, hyperparameter search, attention heatmaps
- `skeleton_hw2_q2/skeleton_code/` — base skeleton provided by course
-  report shows results, drawing comparisons and interpretations, as well as indicating future improvements
