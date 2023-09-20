# AWS Sagemaker image classification and ML Workflow using Lambda and Step Functions

This project was completed as part of the Udacity AWS Machine Learning Engineering Nanodegree. 

## Overview
The context was that we had been hired to build an ML Workflow for a delivery company that could be used to route delivery drivers to the correct loading bay and orders by way of training an ML image classifier that could detect what kind of vehicle the drivers had. In this way, workers on bicycles could be assigned to closer orders while motorcyclists could take the further ones, thus helping to optimize operations. 

The project consists of the following steps.

1. Data staging
2. Model training and deployment
3. Lambdas and step function workflow
4. Testing and evaluation

## Dataset

We used a subset of the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank" rel="noopener">CIFAR dataset to train the model on</a>