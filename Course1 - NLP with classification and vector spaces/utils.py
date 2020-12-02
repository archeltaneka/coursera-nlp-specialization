import re
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples

def preprocess_tweet(tweet):
    
    # copy over the tweet input
    new_tweet = tweet
    # remove links and URLS
    new_tweet = re.sub(r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', tweet)
    # remove hashtags
    new_tweet = re.sub(r'#', '', new_tweet)
    # remove old style retweets
    new_tweet = re.sub(r'^RT[\s]+', '', new_tweet)
    
    stemmer = PorterStemmer()
    eng_stopwords = stopwords.words('english')
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    tweet_tokens = tokenizer.tokenize(new_tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if word in eng_stopwords:
            continue
        stem = stemmer.stem(word)
        stem = stem.lower()
        
        tweets_clean.append(stem)
    
    return tweets_clean


def build_freqs(features, labels):
    
    freq = {}
    
    for (x, y) in zip(features, labels):
        x = preprocess_tweet(x)
        for word in x:
            pair = (word, y)

            if pair not in freq:
                freq[pair] = 1
            else:
                freq[pair] += 1

    return freq

def lookup(freqs, word, label):
    
    n = 0
    pair = (word, label)
    
    if pair in freqs:
        n = freqs[pair]
        
    return n