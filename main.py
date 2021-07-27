
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv
from time import time
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
filename ="SPECIFY INPUT FILE DESTINATION HERE"

def urls_list(file):
    with open(file,newline='') as f:
      reader =csv.reader(f)
      data =[row for row in reader]
    return data

def create_corpus(urls_list):
    corpus =[]
    for list in urls_list:
            doc =scrape(list[0])
            corpus.append(doc)
    return corpus


def scrape(url):
    #Use Beautiful Soup to retirieve all contents of url
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    print("scraping ",url)

    # Find all of the text between paragraph tags and strip out the html
    blacklist = [
        'style',
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script'
    ]
    para =""
    text_elements = [t for t in soup.find_all(text=True) if t.parent.name not in blacklist]
    for possible_paragraph in text_elements:
        if len(possible_paragraph) > 200:# Processes paragraphs (>200 characters)
            para+=possible_paragraph

    return para

def tfidf_dataframe(corpus):
    # create tf-idf features
    print("Extracting tf-idf features")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       stop_words='english')
    t0 = time()
    X = tfidf_vectorizer.fit_transform(corpus)
    print("done in %0.3fs." % (time() - t0))
    features =tfidf_vectorizer.get_feature_names()
    tfidf_vect_df = pd.DataFrame(X.todense(), columns=tfidf_vectorizer.get_feature_names())

    return tfidf_vect_df, features

def create_target_feature(df,list_of_target_names):

    for target_name in list_of_target_names:
        mean =df[target_name].mean()
        print("Mean of ",target_name, "is ",mean)
        df[target_name+"_score"]=[x if x>mean else -1 for x in df[target_name]]
        df["target"]=[target_name if x!=-1 else "other" for x in df[target_name+"_score"]]

    for ind in df.index:
        max_all =max(df['technology_score'][ind],df['markets_score'][ind],df['nft_score'][ind],df['ethereum_score'][ind],df["bitcoin_score"][ind],df["blockchain_score"][ind],df["business_score"][ind] )

        if max_all==-1:
            df.loc[ind,"target"]=0
        elif df["bitcoin_score"][ind]==max_all:
            df.loc[ind,"target"]=11
        elif df["markets_score"][ind]==max_all:
            df.loc[ind,"target"]=12
        elif df["ethereum_score"][ind]==max_all:
            df.loc[ind,"target"]=13
        elif df["technology_score"][ind]==max_all:
            df.loc[ind,"target"]=14
        elif df["nft_score"][ind]==max_all:
            df.loc[ind,"target"]=15
        elif df["blockchain_score"][ind]==max_all:
            df.loc[ind,"target"]=16
        elif df["business_score"][ind]==max_all:
            df.loc[ind,"target"]=17

    df.to_csv("df_new_labels.csv")
    return df

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices(
      (
          dict(dataframe), labels.astype(int)
      ))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# main script.
if __name__ == '__main__':
    #read urls from the file and store in a list
    urlslist =urls_list(filename)

    #scrape text data from the urls in the list and create a corpus
    corpus =create_corpus(urlslist)

    #from corpus, create a pandas dataframe with TFIDF frequencies
    X_df, features =tfidf_dataframe(corpus)
    print(features)
    #specify categories for categorization
    targets =['bitcoin','technology','markets','ethereum','nft','blockchain','business']
    #create target feature based on the TFIDF frequencies of the associated terms
    df =create_target_feature(X_df,targets)

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    for feature_batch, label_batch in train_ds.take(1):
        print('Every feature:', list(feature_batch.keys()))

        print('A batch of targets:', label_batch)

    """
    training_df: pd.DataFrame = df
    
    x_train = np.asarray(training_df[features].values).astype(np.float32)
    y_train =training_df['target'].astype(int)
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(x_train, tf.float32),
                tf.cast(training_df['target'].values, tf.int32)
            )
        )
    )
    """
    feature_columns=[]
    # numeric col
    for header in features:
        if header in ["bündchen","condé",'grigòlo','gül','kármán']: #stray words that are not definted in TF scope, let's eliminate them
            print(header)
        else:
            feature_columns.append(feature_column.numeric_column(header))

    # Now that we have defined our feature column, we will use a DenseFeatures layer to input it to our Keras model.
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # create the model
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.1),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    model.fit(train_ds,
              validation_data=test_ds,
              epochs=10)

    model.summary()

    print("Evaluate on test data")
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)




"""
Key Point: You will typically see best results with deep learning with much larger and more complex datasets. 
When working with a small dataset like this one, we recommend using a decision tree or random forest as a strong baseline. The goal of this tutorial is not to train an accurate model, but to demonstrate the mechanics of working with structured data, so you have code to use as 
a starting point when working with your own datasets in the future.

"""


