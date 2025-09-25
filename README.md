This repository contains a machine learning project predicting Ames, Iowa home sale prices. The dataset spans 2006–2010 with features describing lot, structure, quality, neighborhood, house condition, etc. I use a chronological split—2006–2009 for training and 2010 for testing—to simulate a real forecasting task and avoid time leakage. Model selection relies on repeated K-fold cross-validation with grid search, randomized search, and Optuna; LightGBM is the best performer with an R² of 0.942 on the 2010 holdout.

data_description.txt provides concise definitions for every field in the Ames dataset (column names, meanings, and value encodings).

Data_Cleaning_Preprocessing.ipynb prepares the inputs used by all models. It handles missingness (for example, imputing LotFrontage from the training subset only to prevent leakage), converts ordinal features to ordered categories, removes perfectly collinear fields, excludes a small number of problematic observations, and applies light feature engineering. The cleaned outputs from this notebook are consumed by every modeling notebook.

EDA.ipynb explores the data and complements the modeling. It examines the distribution of SalePrice (and its log transform), tests whether YrSold meaningfully affects prices using ANOVA, surfaces seasonality in sales by quarter, and runs a post-model sanity check to confirm that the three most important predictors from the winning model behave sensibly. 

Model_Evaluation.ipynb compiles the key scores across models, emphasizing the test R² and R² from cross-validation.

The modeling notebooks each focus on a single algorithm and follow the same transparent workflow. Multiple_Linear_Regression.ipynb establishes a baseline OLS and also explores regularized models (Ridge, Lasso, Elastic Net). GradientBoosting.ipynb, Random_Forest.ipynb, LightGBM.ipynb, and CatBoost.ipynb train the corresponding tree-based models with tuned hyperparameters, evaluate via repeated cross-validation on 2006–2009, retrain on the full training period, and report the 2010 test result. Each notebook also uses the SHAP package to identify impactful predictors and feature interactions.

To run the project, start with Data_Cleaning_Preprocessing.ipynb to generate the cleaned data file. Then open any model notebook and execute the cells to train, tune, and evaluate. Model_Evaluation.ipynb summarizes results once individual runs are complete. EDA.ipynb can be run at any point and it reads the cleaned data produced by Data_Cleaning_Preprocessing.ipynb.

The Ames Housing dataset was curated by Professor Dean De Cock (see his [paper](https://jse.amstat.org/v19n3/decock.pdf) for details). This work relies on the open-source ecosystems around scikit-learn, SHAP, LightGBM, CatBoost, Optuna, pandas, seaborn, and matplotlib. I can be reached at [erinyyu3@gmail.com](mailto:erinyyu3@gmail.com) for comments or questions.
