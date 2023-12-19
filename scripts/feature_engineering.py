import click
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import os

nltk.download('stopwords')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
stopwords_list = list(stopwords.words('english'))

def get_length_in_words(text):
    """
    Returns the length of the text in words.

    Parameters:
    ------
    text: (str)
    the input text

    Returns:
    -------
    length of tokenized text: (int)
    """
    return len(nltk.word_tokenize(text))

@click.command()
@click.option('--train-data', type=str, help="Path to train data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
def main(train_data, test_data, data_to):
    """
    Preprocesses the train and test data and saves them as CSV files.

    This function loads the train and test data, applies TF-IDF vectorization to the 'comment_text' column, 
    adds a new column 'n_words' which contains the length of the text in words, and saves the 
    processed data as CSV files.

    Parameters:
    ------
    train_data: str
        The path to the train data file.

    test_data: str
        The path to the test data file.

    data_to: str
        The path to the directory where the processed data will be saved.

    Returns:
    -------
    None
    """
    # Load the train data
    train_df = pd.read_csv(train_data)
    preprocess_data(train_df, data_to, "train.csv")

    # Load the test data
    test_df = pd.read_csv(test_data)
    preprocess_data(test_df, data_to, "test.csv")

def preprocess_data(df, data_to, filename):
    x = df.comment_text

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        max_features=10000,
        stop_words=stopwords_list
    )
    word_vectorizer.fit(x)
    x_train_word_features = word_vectorizer.transform(x)

    df = df.assign(n_words=x.apply(get_length_in_words))

    # Save the dataframe as a CSV file
    df.to_csv(os.path.join(data_to, filename), index=False)


if __name__ == "__main__":
    main()
