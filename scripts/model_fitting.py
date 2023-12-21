import click
import os
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import cross_validate
from joblib import dump

@click.command()
@click.option('--original-train', type=str, help="Path to original training data")
@click.option('--pipeline-to', type=str, help="Path to directory where the model pipeline object will be written to")
@click.option('--result-to', type=str, help="Path to directory where the result will be written to")
@click.option('--seed', type=int, help="Random seed", default=1)



def main(original_train, pipeline_to, result_to, seed):
    '''Main function that fits the pipelines, performs cross-validation, and saves the results and the pipelines.'''

    # Set the random seed and the output format of scikit-learn
    np.random.seed(seed)
    #set_config(transform_output="pandas")

    # Load the original training data
    train_df = pd.read_csv(original_train)

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', CountVectorizer(stop_words="english", max_features=800), 'comment_text'),
            ('num', StandardScaler(), ['n_words', 'vader_sentiment'])
        ]
    )

    # Define the classifiers
    dtree = DecisionTreeClassifier(max_depth=50, random_state=seed)
    lgbm = LGBMClassifier(n_estimators=100, random_state=seed)
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=seed)

    # Define the pipelines
    pipelines = {
        'decision_tree': Pipeline([('preprocessor', preprocessor), ('classifier', dtree)]),
        'lightgbm': Pipeline([('preprocessor', preprocessor), ('classifier', lgbm)]),
        'neural_network': Pipeline([('preprocessor', preprocessor), ('classifier', nn)]),
    }

    # Define your input features and target labels
    X = train_df[['comment_text', 'n_words', 'vader_sentiment']]
    y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    # Define the scoring metrics
    scoring = ['roc_auc', 'f1']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Function to calculate ROC AUC and F1 score
    def calculate_metrics(true_labels, predicted_labels):
        roc_auc = roc_auc_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        return {'roc_auc': roc_auc, 'f1': f1}

    # Perform cross-validation for each pipeline and each label
    results = []
    for name, pipeline in pipelines.items():
        for class_name in y.columns:
            # Fit the pipeline on the training data
            pipeline.fit(X_train, y_train[class_name])

            # Evaluate the model on the validation set
            y_val_pred = pipeline.predict(X_val)

            # Calculate evaluation metrics
            val_metrics = calculate_metrics(y_val[class_name], y_val_pred)

            results.append({
                'model': name,
                'label': class_name,
                'val_roc_auc': val_metrics['roc_auc'],
                'val_f1': val_metrics['f1'],
            })

            # Save the pipelines
            os.makedirs(pipeline_to, exist_ok=True)
            dump(pipeline, os.path.join(pipeline_to, f"{name}_{class_name}.joblib"))

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results
    os.makedirs(result_to, exist_ok=True)
    results_df.to_csv(os.path.join(result_to, 'validation_result.csv'), index=False)



if __name__ == "__main__":
    main()