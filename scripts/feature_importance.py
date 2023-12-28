import click
import pandas as pd
import altair as alt
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from joblib import dump

@click.command()
@click.option('--original-train', type=str, help="Path to original training data")
@click.option('--pipeline-to', type=str, help="Path to directory where the model pipeline object will be written to")
@click.option('--result-to', type=str, help="Path to directory where the result will be written to")
@click.option('--seed', type=int, help="Random seed", default=1)
def main(original_train, pipeline_to, result_to, seed):
    # Load your data
    train_df = pd.read_csv(original_train)

    X = train_df[['comment_text', 'n_words', 'vader_sentiment']]
    y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', CountVectorizer(stop_words="english", max_features=800), 'comment_text'),
            ('num', StandardScaler(), ['n_words', 'vader_sentiment'])
        ]
    )

    # Initialize an empty DataFrame for storing feature importances
    all_importances = pd.DataFrame()

    # Define the pipeline
    pipelines = {
        'decision_tree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=seed))
        ])
    }

    # Loop over each target class
    for class_name in y.columns:
        # Fit the decision tree pipeline
        pipelines['decision_tree'].fit(X, y[class_name])

        # Get feature names from CountVectorizer
        text_features = preprocessor.named_transformers_['text'].get_feature_names_out()
        # Combine all feature names
        feature_names = list(text_features) + ['n_words', 'vader_sentiment']

        # Get the classifier
        classifier = pipelines['decision_tree'].named_steps['classifier']

        # Get feature importances
        importances = classifier.feature_importances_

        # Create a DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Class': class_name
        })

        # Append to the all_importances DataFrame
        all_importances = pd.concat([all_importances, importance_df], ignore_index=True)

        # Save the pipeline
        dump(pipelines['decision_tree'], f'{pipeline_to}/{class_name}_pipeline.joblib')

    # Loop over each class
    for class_name in y.columns.unique():
        # Filter the DataFrame for the current class
        class_importances = all_importances[all_importances['Class'] == class_name]

        # Plot the feature importances with Altair
        chart = alt.Chart(class_importances).mark_bar().encode(
            x='Importance:Q',
            y=alt.Y('Feature:N', sort='-x'),
            tooltip=['Feature', 'Importance', 'Class']
        ).transform_filter(
            alt.datum.Importance >= 0.01  # Filter to include only the top features
        ).properties(
            title=f'Top Feature Importances for {class_name}'
        )

        # Save the chart
        chart.save(f'{result_to}/{class_name}_importances.html')

if __name__ == '__main__':
    main()
