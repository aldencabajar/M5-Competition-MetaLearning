# Meta-Learning Solution for the M5 Competition (Uncertainty)

## Introduction
The M5 Competition is the 5th Iteration of the M series of Forecasting competitions from the University of Nicosia. For this iteration, Walmart sales at different scales (from item-level to overall) are to be predicted, for a total of about 40,000 different time series to be predicted. The competition was divided into two parts: Accuracy and Uncertainty. For the accuracy part, point estimates are to be evaluated while uncertainty quantiles will be evaluated for the uncertainty part.

## Solution
For this competition, I used the **FFORMA** (short for Feature-based forecast model averaging)
meta-learning approach as described in  
[Montero Manso et al.'s paper](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300895).
This method aims to train a meta-learner that learns the appropriate weights to be given 
to each individual model forecast. Since we are also interested on the uncertainty quantiles, 
we want to optimize the forecast loss: 

<p align="center"><img src="svgs/644c43014a081a4d89adeaa4dc8dded7.svg?invert_in_darkmode" align=middle width=126.1344216pt height=47.60747145pt/></p>

Where <img src="svgs/c0984ec57b5102d9ca962e9c3e67d4a8.svg?invert_in_darkmode" align=middle width=24.695747999999988pt height=22.831056599999986pt/> is the loss at time step <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, for individual model <img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> 
and <img src="svgs/d2392d2717cb1369b798f978cff33235.svg?invert_in_darkmode" align=middle width=42.88066694999999pt height=24.65753399999998pt/> is the weight function with predictors <img src="svgs/c9c53a99901c4a67544997f70b0f01bc.svg?invert_in_darkmode" align=middle width=18.696821549999992pt height=22.465723500000017pt/>. 

The meta-learner that was used was a tree-based gradient boosting 
algorithm. See paper for details on the implementation.


