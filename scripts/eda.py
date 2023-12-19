# eda.py
# author: Bill Wan
# date: 2023-12-19

import click
import os
import altair as alt
import pandas as pd

@click.command()
@click.option('--training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")

def main(training_data, plot_to):
    '''
    This script performs exploratory data analysis (EDA) on the provided training data.
    It generates two plots: one depicting the distribution of comment text length and
    another showing the counts per class in the dataset.

    Parameters:
    --training-data (str): Path to the processed training data in CSV format.
    --plot-to (str): Path to the directory where the generated plots will be saved.

    Usage:
    python eda.py --training-data path/to/training_data.csv --plot-to path/to/plots_directory
    '''

    train_df = pd.read_csv(training_data)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i in labels:
        train_df = train_df.loc[train_df[i] >= 0]

    alt.data_transformers.enable('default', max_rows=None)

    # Distribution of Comment Text Length
    comment_len = train_df['comment_text'].str.len()
    len_chart = alt.Chart(comment_len.reset_index()).mark_bar().encode(
        alt.X("comment_text:Q", bin=alt.Bin(maxbins=20), title="Comment Length"),
        alt.Y("count()", title="Frequency"),
    ).properties(
        title="Distribution of Comment Text Length",
        width=400,
        height=300,
    )

    # Counts Per Class
    labels_per_comment = train_df[labels].sum(axis=1)
    train_df['is_clean'] = 0
    train_df.loc[labels_per_comment == 0, 'is_clean'] = 1

    label_counts = train_df[labels + ['is_clean']].sum()
    label_counts_df = pd.DataFrame({'Label': label_counts.index, 'Count': label_counts.values})

    class_dist = alt.Chart(label_counts_df).mark_bar().encode(
        x='Label',
        y='Count',
        color=alt.Color('Label:N'),
        tooltip=['Label', 'Count']
    ).properties(
        title="Counts Per Class",
        width=400,
        height=300
    )

    train_df.drop('is_clean', axis=1, inplace=True)

    # Save Plots
    os.makedirs(plot_to, exist_ok=True)
    len_chart.save(os.path.join(plot_to, "comment_text_length_distribution.html"), embed_options={'renderer': 'svg'})
    class_dist.save(os.path.join(plot_to, "class_distribution.html"), embed_options={'renderer': 'svg'})

if __name__ == '__main__':
    main()
