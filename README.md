# RidgeRegressionProject
This is the repository of the Ridge Regression project for the Statistical Methods for Machine Learning course

The report is in the file [Report.pdf](https://github.com/fratorgano/RidgeRegressionProject/blob/main/Report.pdf)

The file [ridgeReg.py](https://github.com/fratorgano/RidgeRegressionProject/blob/main/ridgeReg.py) contains the implementation of the Ridge Regression predictor

Main experiments:
* [ridge_numerical.ipynb](https://github.com/fratorgano/RidgeRegressionProject/blob/main/ridge_numerical.ipynb) - train and test the ridge regressor using only the numerical part of the dataset
* [Encoding.ipynb](https://github.com/fratorgano/RidgeRegressionProject/blob/main/Encoding.ipynb) - analyze the possible different encoding types and check performance
* [Encoding_alternative.ipynb](https://github.com/fratorgano/RidgeRegressionProject/blob/main/Encoding_alternative.ipynb) - analyze the possible different encoding types and check performance (with a different dataset preprocessing)
* [ridge_all.ipynb](https://github.com/fratorgano/RidgeRegressionProject/blob/main/ridge_all.ipynb) - train and test the ridge regressor using the whole dataset and try to find best value for parameters

Reproducibility: 
* All the data and graphs reported were generated using Python 3.10.12 on Ubuntu 22.04 using WSL2 on Windows 11
* All the python packages used, along with their versions, are available in the file [requirements.txt](https://github.com/fratorgano/RidgeRegressionProject/blob/main/requirements.txt)
* All the random seeds, where needed, were set to 1 to ensure reproducibility of the results
