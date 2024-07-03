How does RASCAL work?
============================

.. _Analog Method:

The Analog Method
--------------------
RASCAL is based on a Empirical Statistical Downscaling method know as Analog Models/Method or Weather Generators. This is a widely used technique in climate research. It is based on the premise that large-scale atmospheric conditions tend to produce comparable
local weather patterns, allowing the prediction of local conditions for a day without real-time observations. This is done by
identifying an analog day from General Circulation Models (GCMs), such as reanalyses, and assigning its local conditions.
This approach allows the study of climate variability over an extended time frame, providing valuable perspectives on long-term patterns and connections between different geographic locations, while also incorporating important local factors into the
analysis.
The analog method is nonlinear technique that relies on the identification of strong statistical relationships between two
fields: the predictor variable extracted from GCM products, and the predictand variable obtained from local historical observations. To predict an atmospheric feature (the predictand) for a given day, this method searches for the day with the most similar predictor field in the historical record and uses its atmospheric features to make a prediction, allowing the reconstruction of
missing data points.


.. image:: 
  ../images/analog.png

.. _PCA:

Principal Component Analysis
------------------------------

To incorporate the relationship between large-scale meteorological patterns and local weather, the analog method is often
combined with Principal Component Analysis (PCA). The PCA reduces the high dimensionality of the atmospheric phase
space by generating an orthogonal basis of vectors that represent the main directions of variability. As a result, only a limited
set of coefficients, called principal components (PCs), are required to represent the atmospheric state.


You can read in `(González-Cervera & Durán, 2024) <https://doi.org/10.5194/egusphere-2024-958>`_ more details about the methodology and performance of the model.

.. image:: 
  ../images/flux_diagram.png
