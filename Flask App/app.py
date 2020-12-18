from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import tweepy

model_pipeline = load("model.joblib")

api_key = "<Key>"
# api secret key
api_secret_key = "<Key>"
# access token
access_token = "<Key>"
# access token secret
access_token_secret = "<Key>"

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)

tweets_list = []
result_sentiment = ""
similar_text = ""
similar_perc = ""
less_similar_text = ""
less_similar_perc = ""
least_similar_text = ""
least_similar_perc = ""
result2 = ""
positive_count = 0
negative_count = 0


def get_related_tweets(topic):
    count = 50
    lang = 'en'
    try:
        tweets_list.clear()
        for tweet in api.search(q=topic, count=count, lang=lang):
            tweets_list.append(tweet.text)
    except BaseException as e:
        print('failed on_status,', str(e))


def get_sentiment(query):
    result = model_pipeline.predict([query])
    if str(result[0]) == "1":
        return "Positive"
    return "Negative"


def get_general_sentiment(topic):
    get_related_tweets(topic)
    positives = 0
    negatives = 0
    running_total = 0
    for sentence in tweets_list:
        if get_sentiment(sentence) == "Positive":
            running_total += 1
            positives += 1
        else:
            running_total -= 1
            negatives += 1
    if running_total == 0:
        return "Neutral", positives, negatives
    if running_total / len(tweets_list) >= 0.5:
        return "Overwhelmingly Positive", positives, negatives
    if running_total / len(tweets_list) > 0:
        return "Generally Positive", positives, negatives
    if running_total / len(tweets_list) > -0.5:
        return "Generally Negative", positives, negatives
    if running_total / len(tweets_list) <= -0.5:
        return "Overwhelmingly Negative", positives, negatives


def get_similar(index):
    tfidf_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), stop_words=ENGLISH_STOP_WORDS)
    vectorized_list = tfidf_vec.fit_transform(tweets_list)
    cos_arr = cosine_similarity(vectorized_list[index], vectorized_list)
    i = np.unique(cos_arr)[-2]
    j = np.unique(cos_arr)[-3]
    k = np.unique(cos_arr)[-4]
    a = np.where(cos_arr == i)[1][0]
    b = np.where(cos_arr == j)[1][0]
    c = np.where(cos_arr == k)[1][0]
    return tweets_list[a], str(round(float(i)*100, 2))+'%', tweets_list[b], str(round(float(j)*100, 2))+'%', tweets_list[c], str(round(float(k)*100, 2))+'%'


app = Flask(__name__, template_folder="pages")


@app.route('/')
def home():
    return render_template('forms/Home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        query = request.form['search']
        # return get_sentiment(query)
        # return get_general_sentiment(query)
        result_sentiment, positive_count, negative_count = get_general_sentiment(query)
        return render_template('forms/Home.html', result=result_sentiment, tweets=tweets_list, pos = str(positive_count), neg = str(negative_count))


@app.route('/similar/', methods=['POST'])
def similar():
    try:
        query = request.form['tweets']
        query_index = tweets_list.index(query)
        # return get_similar(query_index)
        similar_text, similar_perc, less_similar_text, less_similar_perc, least_similar_text, least_similar_perc = get_similar(query_index)
        return render_template('forms/Home.html', result=result_sentiment, tweets=tweets_list, similar=similar_text,
                               perc=similar_perc, similar2=less_similar_text, perc2=less_similar_perc, 
                               similar3=least_similar_text, perc3=least_similar_perc)
    except ValueError:
        return "Invalid Selection"


@app.route('/manual/', methods=['POST', 'GET'])
def check_review():
    if request.method == 'POST':
        query = request.form['search2']
        res = get_sentiment(query)
        return render_template('forms/Home.html', result2=res)


if __name__ == '__main__':
    app.run(debug=True)
