# Topic-Detection-unsupervised-from-urls
This program scrapes text from a list of urls (in csv) and creates TFIDF features from every term and puts it into a dataframe. It also creates the 'target' variable from the these features. We then use TF to create a basic network to classify. 
Topic detection from urls 

**Packages you need to install: **
pandas, tensorflow, bs4, numpy, scikit-learn, time, csv

**commands to install them:**

pip install pandas

pin install -U scikit-learn

pip install bs4

pip install --upgrade tensorflow

if you face issues with TF installation, follow https://www.tensorflow.org/install/pip

Input:
Please input the destination file for your list of urls (in csv) as filename. 

This programs scrapes "significant" paragraphs from a list of provided urls. It uses Beautiful Soup to do this. Once that is done, the paragraphs are collated into a list. We then run the list of paragrpahs through a function defined as tfidf_dataframe. This function creates a pandas dataframe by capturing TFIDF of all the terms in these paragraphs (we will call each url's content as a document, so as not to confuse paragraphs as each url would contain multiple paragraphs). Scikit's TFIDF vectoriser is used for this purpose, and we can see that it captures relvant term frequencies, rather that just counting ALL the frequencies of all words. 

"""
Whats' TFIDF? TF-IDF stands for Term Frequency — Inverse Document Frequency
Here is a blog that deep-dives into TFIDF and how it is used in information retrieval: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
"""

we are also able to remove ALL stop words, so we will not be taxing any classifier by making them run through un-necessary words such as "this" and "that". We are hoping that the classifier looks at words such as "bitcoin", "ethereum", 'BTC', "crypto" etc. 

Eg: the TFIDF score for the term 'wallet' and 'bitcoin' for the first url
wallet	            
0.07957242026609590	

bitcoin
0.18931356854040600


The task of classifying documents into multiple classes (meaning it's not a binary classification task) is a hard problem. It becomes extremely complex, when we add the constraint of non-supervision, meaning we have no training dataset which we can feed the classifier to "learn from".

So a "creative" way of solving this was to create the training data set using the word-frequencies (that we can extract using the TFIDF vectoriser). We can assume that this approach is rather good, but the only way to fully validate this approach is to actually manually compare results (which wasn't done).

so for a term like "bitcoin", we would calculate the TFIDF frequencies for all the documents. The documents that have "significant" presence of the term would have higher TFIDF scores. so for each category, we can calculate a cumulative score for each document. 



so for document A, we have: (category + "score", labeling to make it easier to understand)

bitcoin_score
0.18907582749645400

technology_score (no presence)
-1.0

trading_score
0.34438080201746400

art_score
0.028341746599359800

crypto_score
0.2914181223972970

so the score for the "bitcoin" occurences was lesser than "trading" and "crypto", so we can file this document under "trading". I followed this scoring approach for all the documents. I also had a label for "others"

Our target column looked something like this: (a cursory look revealed that a lot of the documents belonged to the category "others")

trading
others
bitcoin
technology
crypto
trading
technology
bitcoin
others
crypto
bitcoin
crypto
others
technology
technology
crypto
crypto
bitcoin

Once this was done, we created a TF dataset from the dataframe, split the training dataset into test sets and validation sets. We only have 800 or so documents for this training, which is extremely less for the segmentation. 

With an approach of 783 train examples & 88 test examples, we have obtained pretty bad performance. 

Copied from TF blog is: 
"""
**Key Point:** You will typically see best results with deep learning with much larger and more complex datasets. 
When working with a small dataset like this one, we recommend using a decision tree or random forest as a strong baseline. The goal of this tutorial is not to train an accurate model, but to demonstrate the mechanics of working with structured data, so you have code to use as a starting point when working with your own datasets in the future.

"""

**How to improve:**
1. More data. Think 10,000+ documents.
2. Some supervision. Even 20-30% can improve performance - we can use the above approach as well to get more supervision
3. We can use raw text in addition to the TFIDF frequecies as addiitonal features
4. We can also think of somehow emodying relationships in text as features. For eq. here we consider each word as independent of each other, we can figure out a way to also capture and quantify relationships. 
