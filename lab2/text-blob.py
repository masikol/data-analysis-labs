from textblob import TextBlob
import tweepy as twit
import preprocessor as prep

if __name__ == "__main__":
    twitter_api_key = '9bjRdeeugv2FkMTPQmcDRZAdv'
    twitter_api_secret = 'z0iaQR7xoejmGi8ElvA65C0hs6M9IhMgQCWSvy5sRjx3iYp02E'
    twitter_bearer_token = f'AAAAAAAAAAAAAAAAAAAAAKi4WAEAAAAAQG65Ns9pXb%2F2SuQXCMEjE5dgGck%3D37luuxlRj1KLBZWBaPQlZ8BX1fauKDluH57vbJDIdahDFF7rte'
    access_token = '927121523739103232-j1LdP7Jj9CsWvK84byv4TLLStgLMj8p'
    access_token_secret = 'oTYCNp4KYB5yvKYgAkQPiX6uJ3NeWh6dRZ2D0UsY84l2A'
    ilon_musk_id = 44196397
    britney_spears_id = 16409683

    # подключаемся к твиттеру по АПИ
    client = twit.Client(twitter_bearer_token, twitter_api_key, twitter_api_secret, wait_on_rate_limit=True)
    
    print('INVESTIGATING Bretney Spears\' 5 tweets')
    tweets = client.get_users_tweets(britney_spears_id, exclude=['retweets', 'replies'], max_results=5)
    for index, tweet in enumerate(tweets.data):
        text = prep.clean(tweet.text)
        # TextBlob это АПИ для проведения задач NLP, таких как семантический анализ, 
        # классификация, перевод и т.д. # без необходимости разбираться в реализации
        blob = TextBlob(text)
        print(str(index+1) + '. [' + ('Positive' if blob.sentiment[0] > 0 else 'Neutral' if blob.sentiment[0] == 0 else 'Negative') + '] (clean text)' + text)

    print('INVESTIGATING Ilon Musk\' 5 tweets')
    tweets = client.get_users_tweets(ilon_musk_id, exclude=['retweets', 'replies'], max_results=5)
    for index, tweet in enumerate(tweets.data):
        text = prep.clean(tweet.text)
        blob = TextBlob(text)
        print(str(index+1) + '. [' + ('Positive' if blob.sentiment[0] > 0 else 'Neutral' if blob.sentiment[0] == 0 else 'Negative') + '] (clean text)' + text)