import click
import nltk
import pandas as pd
import os
import nlpaug.augmenter.word as naw
import altair as alt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_length_in_words(text):
    return len(nltk.word_tokenize(text))

def augment_data(df, aug1, aug2):
    df['augmented_text1'] = df['comment_text'].apply(lambda x: aug1.augment(x))
    df['augmented_text2'] = df['comment_text'].apply(lambda x: aug2.augment(x))
    
    df_aug1 = df.copy()
    df_aug1['comment_text'] = df['augmented_text1']
    
    df_aug2 = df.copy()
    df_aug2['comment_text'] = df['augmented_text2']

    return pd.concat([df, df_aug1, df_aug2])

@click.command()
@click.option('--train-data', type=str, help="Path to train data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--plot-to', type=str, help="Path to directory where processed plot will be written to")

def main(train_data, test_data, data_to, plot_to):
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    # Separate data based on labels
    df_toxic = train_df.loc[train_df['toxic'] == 1]
    df_sevtox = train_df.loc[train_df['severe_toxic'] == 1]
    df_obs = train_df.loc[train_df['obscene'] == 1]
    df_threat = train_df.loc[train_df['threat'] == 1]
    df_insult = train_df.loc[train_df['insult'] == 1]
    df_hate = train_df.loc[train_df['identity_hate'] == 1]
    df_clean = train_df[train_df[labels].sum(axis=1) == 0]

    # Define augmentation methods
    aug1 = naw.SynonymAug(aug_src='wordnet')
    aug2 = naw.RandomWordAug(action='swap')

    # Augment data based on labels
    df_toxic = augment_data(df_toxic, aug1, aug2)
    df_sevtox = augment_data(df_sevtox, aug1, aug2)
    df_obs = augment_data(df_obs, aug1, aug2)
    df_threat = augment_data(df_threat, aug1, aug2)
    df_insult = augment_data(df_insult, aug1, aug2)
    df_hate = augment_data(df_hate, aug1, aug2)

    # Concatenate augmented and clean data
    aug_df = pd.concat([df_toxic, df_sevtox, df_obs, df_threat, df_insult, df_hate])
    aug_df = aug_df.drop(columns=['augmented_text1', 'augmented_text2'])
    train_df = pd.concat([aug_df, df_clean])
    train_df['comment_text'] = train_df['comment_text'].astype(str)

    # Sentiment analysis
    sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    
    train_df = train_df.assign(n_words=train_df['comment_text'].apply(get_length_in_words))
    test_df = test_df.assign(n_words=test_df["comment_text"].apply(get_length_in_words))
    train_df = train_df.assign(vader_sentiment=train_df["comment_text"].apply(lambda x: sid.polarity_scores(x)["compound"]))
    test_df = test_df.assign(vader_sentiment=test_df["comment_text"].apply(lambda x: sid.polarity_scores(x)["compound"]))
    train_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    # Save the processed data
    train_df.to_csv(os.path.join(data_to, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "test.csv"), index=False)

    # Create and save the chart
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
    
    os.makedirs(plot_to, exist_ok=True)
    class_dist.save(os.path.join(plot_to, "augmented_class_distribution.html"), embed_options={'renderer': 'svg'})

if __name__ == "__main__":
    main()
