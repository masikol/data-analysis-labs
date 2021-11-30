from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import enum
from textblob import TextBlob
import tweepy as twit
import preprocessor as prep

# слова которые будут игнорированы при построении TF-IDF матрицы по причинам: 
# 1) слово употребляется в слишком большом количестве текстов, 
# 2) слово употребляется в слишком малом количестве текстов.
# взят из обсуждения на форуме одного из датасетов для семантического анализа
stopword =  ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an','and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do','does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here','hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma','me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them','themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre","youve", 'your', 'yours', 'yourself', 'yourselves']

if __name__=="__main__":
    raw_data = pd.read_csv('/home/akhlebko/Workspace/sentiment-python/files/data/training_data.csv', header=None)
    raw_val_data = pd.read_csv('/home/akhlebko/Workspace/sentiment-python/files/data/test_data.csv', header=None)

    print('PREPARING DATA')
    # векторизация текста в матрицу TF-IDF, где вес слова пропорционален частоте употребления
    # в тексте и обратно пропорчионален частоте употребления во всех документах набора данных
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=1000000,stop_words=stopword)
    vect.fit(raw_data[5].values)
    X = vect.transform(raw_data[5].values)
    Y = raw_data[0].values
    print('FITTING')
    # обучение Наивного Классификатора Байеса, который основывается на теореме Байеса,
    # но с допущением того, что все признаки независимые: наличие одного признака 
    # в наборе данных не зависит от наличия других признаков, поэтому он и наивный
    nb_clf = MultinomialNB()
    nb_clf.fit(X, Y)

    # евалуация
    print('EVAL')
    nb_pred = nb_clf.predict(vect.transform(raw_val_data[5].values))
    print('Naive Bayes Accuracy :',accuracy_score(raw_val_data[0].values,nb_pred))


    twitter_api_key = '9bjRdeeugv2FkMTPQmcDRZAdv'
    twitter_api_secret = 'z0iaQR7xoejmGi8ElvA65C0hs6M9IhMgQCWSvy5sRjx3iYp02E'
    twitter_bearer_token = f'AAAAAAAAAAAAAAAAAAAAAKi4WAEAAAAAQG65Ns9pXb%2F2SuQXCMEjE5dgGck%3D37luuxlRj1KLBZWBaPQlZ8BX1fauKDluH57vbJDIdahDFF7rte'
    access_token = '927121523739103232-j1LdP7Jj9CsWvK84byv4TLLStgLMj8p'
    access_token_secret = 'oTYCNp4KYB5yvKYgAkQPiX6uJ3NeWh6dRZ2D0UsY84l2A'
    ilon_musk_id = 44196397
    britney_spears_id = 16409683
    client = twit.Client(twitter_bearer_token, twitter_api_key, twitter_api_secret, wait_on_rate_limit=True)
    
    print('INVESTIGATING Bretney Spears\' 5 tweets')
    tweets = client.get_users_tweets(britney_spears_id, exclude=['retweets', 'replies'], max_results=5)
    for index, tweet in enumerate(tweets.data):
        text = prep.clean(tweet.text)
        result = nb_clf.predict(vect.transform([text]))
        print(str(index+1) + '. [' + ('Positive' if result[0] > 2 else 'Neutral' if result[0] == 2 else 'Negative') + '] (clean text)' + text)

    print('INVESTIGATING Ilon Musk\' 5 tweets')
    tweets = client.get_users_tweets(ilon_musk_id, exclude=['retweets', 'replies'], max_results=5)
    for index, tweet in enumerate(tweets.data):
        text = prep.clean(tweet.text)
        result = nb_clf.predict(vect.transform([text]))
        print(str(index+1) + '. [' + ('Positive' if result[0] > 2 else 'Neutral' if result[0] == 2 else 'Negative') + '] (clean text)' + text)
