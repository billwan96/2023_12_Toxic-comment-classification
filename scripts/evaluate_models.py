import click
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import joblib

@click.command()
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--model-dir', type=str, help="Directory containing trained models")
@click.option('--output-dir', type=str, help="Directory to save output CSV files")
def main(test_data, model_dir, output_dir):
    # Load test data
    test_df = pd.read_csv(test_data)
    X_test = test_df[['comment_text', 'n_words', 'vader_sentiment']]
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Load the saved pipelines
    pipelines = {}
    for class_name in labels:
        pipeline_path = f'{model_dir}/{class_name}_pipeline.joblib'
        pipelines[class_name] = joblib.load(pipeline_path)

    # Initialize dictionaries for storing evaluation metrics
    roc_auc_scores = {}
    f1_scores = {}
    confusion_matrices = {}

    # Loop over each target class
    for class_name, pipeline in pipelines.items():
        # Apply the saved pipeline to the test data
        y_true = test_df[class_name]
        y_pred = pipeline.predict(X_test)

        # Calculate evaluation metrics
        roc_auc_scores[class_name] = roc_auc_score(y_true, y_pred)
        f1_scores[class_name] = f1_score(y_true, y_pred)
        confusion_matrices[class_name] = confusion_matrix(y_true, y_pred)

        # Save the evaluation metrics to a CSV file
        metrics_df = pd.DataFrame({
            'ROC-AUC': [roc_auc_scores[class_name]],
            'F1 Score': [f1_scores[class_name]]
        })
        metrics_df.to_csv(f'{output_dir}/{class_name}_test_scores.csv', index=False)

        # Save the confusion matrix to a CSV file
        cm_df = pd.DataFrame(confusion_matrices[class_name])
        cm_df.to_csv(f'{output_dir}/{class_name}_test_confusion_matrix.csv', index=False)

if __name__ == '__main__':
    main()

