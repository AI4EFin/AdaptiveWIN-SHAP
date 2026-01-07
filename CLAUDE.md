This is a project that provides a new XAI for time series method that takes into account
changepoint detection (unstationarity) in the data. IT is called:
Adaptive WIN-SHAP or Adapting SHAP to trustworthy window changes.

The basic idea is that we are computing the shap values with a window size that results
from the LPA method developed by Spokoiny in 1998.

Right now we are working on robutstness analysis. The lpa_sensitivity analysis is already done and works
and the visualize_robustness_benchmark is also done and works.

I would like now to have another case of robustness analysis where we modify the parameters
of the data generating process and see how the results change. I would like to start with the
piecewise_ar3 dataset and I would like to test two scenarios where the parameters are closer and further away
in terms of L2 distance and 3 more scenarios with randomised initialisation parameters.

You should save the generated dataset, window sizes and resulting benchmarks so we can visualize them later.
