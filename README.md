# Toxic Comment Classification Challenge

## Overview
This project is part of my ongoing efforts to explore and understand the applications of machine learning and natural language processing in moderating online discussions. The goal is to build a model that can accurately classify toxic comments, which is a crucial step towards creating a safer and more respectful online environment.

The importance of this topic lies in promoting healthy online discussions, preventing cyberbullying, protecting online communities, and improving machine learning and natural language processing models and techniques.

## Dataset
The dataset used in this project is sourced from Kaggle's Toxic Comment Classification Challenge 2018. It includes a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. You can access the dataset [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Methodology
We explored various machine learning and natural language processing techniques to classify the comments. Our best model achieved a notable AUC-ROC score when generalized to unseen data, suggesting high accuracy in toxic comment classification.

## Report
The final report detailing our approach, methodology, and results is available in [this repository](https://github.com/billwan96/2023_12_Toxic-comment-classification/tree/main/report).

## Usage
Setup:
1. Clone the GitHub repository for this project.
2. Create an enviroment from our environment.yml file
```
conda env create -f environment.yml
```

Analysis:
3. At project root, run the following command:
```
# download data and perform eda
python scripts/eda.py --training-data data/raw/train.csv --plot-to results/figures/

# preprocess the data and feature engineering
python scripts/preprocessing_featsengineering.py --train-data data/raw/train.csv --test-data data/raw/test.csv --test-labels data/raw/test_labels.csv --data-to data/processed/ --plot-to results/figures
 
# fitting the models
python scripts/model_fitting.py --original-train data/processed/train.csv --pipeline-to results/models/ --result-to results/tables/

# understand about feature importance of the model
python scripts/feature_importance.py --original-train data/processed/train.csv --pipeline-to results/full_models/ --result-to results/figures/

# evaluate models performance on test data
python evaluate_models.py --test-data data/processed//test.csv --model-dir results/full_models/ --output-dir results/tables/
```

4. (Alternatively you can run all the scripts in one go)
```
# Run the whole analysis
make all

# Remove the analysis
make clean
```

## References
[1]Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. ACM Computing Surveys (CSUR), 51(4), 1-30.

[2]Kaggle. (2018). Toxic Comment Classification Challenge. Retrieved from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

[3]Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 67-73).

## License
The Toxic Comment Classification Challenge report contained herein are licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. The software code is licensed under the MIT License. If re-using/re-mixing, please provide attribution and link to this webpage. See the license file for more information.
