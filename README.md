# BERTgrid: Contextualized Embedding for 2D Document Representation and Understanding
------
**Bertgrid** is based on *Chargrid* by [Kattie et al.(2018)](https://arxiv.org/abs/1809.08799). As *Chargrid*, BERTgrid represents documents as a grid to use as an input. The big difference is that *Chargrid* only embeds each character with ont-hot encoding, **BERTgrid** uses pre-trained BERT model to construct word-piece level grid.

The model is implemented with Pytorch.

[paper](https://arxiv.org/abs/1909.04948)
