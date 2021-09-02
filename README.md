# Learning-Parities-with-Neural-Networks
This code reproduces the results of the paper by Amit Daniely &amp; Eran Malach. It was created as an exam project in the PhD programme of Data Science in SISSA, 2021

Report 
Introduction to NN exam 7/6/2021
Reproduce the results in the paper: Learning Parities with Neural Networks

Binary classification: use 0 even / 1 odd
use k MIST images glued in the same image side by side

1) train a 2-layer NN Relu with 512 neurons			 ReLU
2) Same architecture, keep first layer’s w fixed		ReLU features
3) Linear activation, fixed first layer’s w			Linear features
4) Gaussian initialization, Relu activation, fixed first layer's w	Gaussian Features
5) Glu activation with activation weights fix first layer's w			NTK regime


plot the error during the first 20 epochs: The accuracy is the error measure on the test set, and also monitor the average train loss for each epoch.


Cross entropy loss

128 batch size (suggested in the article)

AdaDelta optimizer (suggested in the article)




