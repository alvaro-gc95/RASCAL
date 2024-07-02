Advanced Tutorials
===================

To assess the skill of the model and to validate the resulting reconstructions, is very useful to divide the input dataset in a "training" and a "testing" datasets. As RASCAL works connecting the large-scale atmospheric patterns with the local weather conditions, there are two different places where you can set this splitting: [1] Split the large scale data when performing the principal component analysis, [2] Split when seeking for analog days in possible analogs and ignorable analogs

**Traing/test split the large-scale data:**
---------------------------------------------------

In RASCAL, the large-scale data information is compressed using Principal Component Analysis. This method allows to find spatial patterns that represent the principal directions of variability (Empirical Orthogonal Functions) and an associated time series that represent the temporal variability of eaf EOF (Principal Components). 



