# Chargrid: Towards Understanding 2D Documents
------
  When we want to do *document understanding* with *deep learning*, the first thing that comes to our mind is to handle it as NLP task. However suddenly, the problem appears. Document understanding is usually 2-dimensional task, different from NLP which solves problem usually in 1-dimensional way. In this paper, they made a **GRID** which can contain not only 1D information. but also 2D ones.
  
  Their goal was to do **Semantic Segmentation** and **Bounding Box Regression** at the same time. The network finds the values which is connected to the key such like the invoice number and total price. It is called semantic segmentation. And the Bounding box regression finds the table area.

<img src="/Chargrid/images/chargrid_1.PNG" width="650" align="center">

  They used encoder-decoder network. The encoder network is consisted of dilated convolution layers and dropout layers. At the decoder network, softmax layer is used for both of semantic segmantation and bounding box regression, however linear layer is only used for bounding box regression.

<img src="/Chargrid/images/chargrid_network.PNG" width="800" align="center">

<img src="/Chargrid/images/chargrid_spartial_distribution.PNG" width="400" align="center">

<img src="/Chargrid/images/chargrid_results.PNG" width="400" align="center">



[paper](https://arxiv.org/pdf/1809.08799.pdf)
