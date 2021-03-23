# Meta-Learning Solution for the M5 Competition (Uncertainty)

## Introduction
The M5 Competition is the 5th Iteration of the M series of Forecasting competitions 
from the University of Nicosia. For this iteration, Walmart sales at different 
scales (from item-level to overall) are to be predicted, for a total of 
42,840 different time series to be predicted. The competition was divided 
into two parts: Accuracy and Uncertainty. For the accuracy part, point estimates 
are to be evaluated while uncertainty quantiles will be evaluated for the uncertainty part.

## Solution
For this competition, I used the **FFORMA** (short for Feature-based forecast model averaging)
meta-learning approach as described in  
[Montero Manso et al.'s paper](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300895).
This method aims to train a meta-learner that learns the appropriate weights to be given 
to each individual model forecast. Since we are also interested on the uncertainty quantiles, 
we want to optimize the forecast loss: 

$$
\sum_{n=1}^{N}{\sum^{M}_{m=1}{\alpha(F_n)l_{nm}}}
$$

Where $l_{nm}$ is the loss at time step $n$, for individual model $m$ 
and $\alpha(F_n)$ is the weight function with predictors $F_n$. 
The meta-learner that was used was a tree-based gradient boosting 
algorithm. I trained one gradient boosting model for each quantile.
See paper and code for more details on the implementation.


## Results

Hyperparameters for the gradient boosting algorithm chosen were the following:

|Parameter| Value |
|---------|--------|
|Max depth | 4 |
| learning rate| 0.114|
|Subsampling prop | 0.9 |
|colsample by tree| 0.6|






