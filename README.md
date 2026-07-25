This is shown as an example of work I can complete, moved from a private repo.

Last commit on this repo was on 2025-08, I have had significant changes to my coding practices since then. Mainly related to moving to wsl, bash commands, and following the `src/` structure more strictly.


# Defaulting risk analysis


The project focusses on analysis and ML for the [https://www.kaggle.com/competitions/home-credit-default-risk/overview](https://www.kaggle.com/competitions/home-credit-default-risk/overview) dataset.

A banking information dataset is analysed with the goal to identify individuals at risk of defaulting. Given that the final call on specific threshold chosen will be given by the credit agency, ROC AUC is used as performance metric throughout the notebooks.

A summary of these and the project itself can be found at:
* [nb0_master.ipynb](notebooks\nb0_master.ipynb)

A final softmax voting classifier (LightGBM GBDT + Logistic Regression) achieved ROC AUC of 0.77109.

Due to to its scope, the project is split across 9 analysis notebooks:
* [nb0_master.ipynb](notebooks\nb0_master.ipynb) - Master overview and project summary
* [nb1_application_cleaning.ipynb](notebooks\nb1_application_cleaning.ipynb) - Application data cleaning and preprocessing
* [nb2_supp_tables_cleaning.ipynb](notebooks\nb2_supp_tables_cleaning.ipynb) - Supplementary tables cleaning
* [nb3_application_eda.ipynb](notebooks\nb3_application_eda.ipynb) - Exploratory data analysis of application data
* [nb4_installments_payments.ipynb](notebooks\nb4_installments_payments.ipynb) - Installments and payments analysis
* [nb5_bureau_balance.ipynb](notebooks\nb5_bureau_balance.ipynb) - Bureau balance table analysis and aggregation
* [nb6_bureau.ipynb](notebooks\nb6_bureau.ipynb) - Bureau table analysis and feature engineering
* [nb7_previous_applications.ipynb](notebooks\nb7_previous_applications.ipynb) - Previous applications analysis and integration
* [nb8_model_tuning.ipynb](notebooks\nb8_model_tuning.ipynb) - Model selection and hyperparameter tuning
* [nb9_model_deployment.ipynb](notebooks\nb9_model_deployment.ipynb) - Model deployment and usage demonstration





# Quick Start Guide

The project can be cloned and results reproduced with the following commands.


``` powershell
git clone https://github.com/TuringCollegeSubmissions/psauci-DS.v2.5.3.4.1
pip install poetry
poetry install --no-root
```

**Note**: The previous commands require python 3.13 installed

The dataset needs to be downloaded from [https://www.kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data), unzipped, and saved into [data/raw_csv](data/raw_csv)



