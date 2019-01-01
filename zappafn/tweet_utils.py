# from https://gist.github.com/timothyrenner/dd487b9fd8081530509c

import imp
import string
import sys

from datetime import datetime

# for running nltk on AWS lambda
# see https://stackoverflow.com/questions/44058239/sqlite3-error-on-aws-lambda-with-python-3  # noqa
sys.modules["sqlite"] = imp.new_module("sqlite")  # noqa
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")  # noqa

# NOTE: need to include downloaded stopwords, otherwise get error:
# "module initialization error: module 'nltk' has no attribute 'data'"
# see https://stackoverflow.com/questions/42382662/using-nltk-corpora-with-aws-lambda-functions-in-python/42389899  # noqa
from nltk.stem.lancaster import LancasterStemmer

# manually extracted from downloaded nltk data
from stopwords import STOPWORDS


# Gets the tweet time.
def get_time(tweet):
    return datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")


# Gets all hashtags.
def get_hashtags(tweet):
    return [tag['text'] for tag in tweet['entities']['hashtags']]


# Gets the screen names of any user mentions.
def get_user_mentions(tweet):
    return [m['screen_name'] for m in tweet['entities']['user_mentions']]


# Gets the text, sans links, hashtags, mentions, media, and symbols.
def get_text_cleaned(tweet):
    text = tweet['text']

    slices = []
    # Strip out the urls.
    if 'urls' in tweet['entities']:
        for url in tweet['entities']['urls']:
            slices += [{'start': url['indices'][0], 'stop': url['indices'][1]}]

    # Strip out the hashtags.
    if 'hashtags' in tweet['entities']:
        for tag in tweet['entities']['hashtags']:
            slices += [{'start': tag['indices'][0], 'stop': tag['indices'][1]}]

    # Strip out the user mentions.
    if 'user_mentions' in tweet['entities']:
        for men in tweet['entities']['user_mentions']:
            slices += [{'start': men['indices'][0], 'stop': men['indices'][1]}]

    # Strip out the media.
    if 'media' in tweet['entities']:
        for med in tweet['entities']['media']:
            slices += [{'start': med['indices'][0], 'stop': med['indices'][1]}]

    # Strip out the symbols.
    if 'symbols' in tweet['entities']:
        for sym in tweet['entities']['symbols']:
            slices += [{'start': sym['indices'][0], 'stop': sym['indices'][1]}]

    # Sort the slices from highest start to lowest.
    slices = sorted(slices, key=lambda x: -x['start'])

    # No offsets, since we're sorted from highest to lowest.
    for s in slices:
        text = text[:s['start']] + text[s['stop']:]

    return text


# Sanitizes the text by removing front and end punctuation,
# making words lower case, and removing any empty strings.
def get_text_sanitized(tweet):
    return ' '.join([w.lower().strip().rstrip(string.punctuation)
                      .lstrip(string.punctuation).strip()
                     for w in get_text_cleaned(tweet).split()
                     if w.strip().rstrip(string.punctuation).strip()])


# Gets the text, clean it, make it lower case, stem the words, and split
# into a vector. Also, remove stop words.
def get_text_normalized(tweet):
    # Sanitize the text first.
    text = get_text_sanitized(tweet).split()

    # Remove the stop words.
    text = [t for t in text if t not in STOPWORDS]

    # Create the stemmer.
    stemmer = LancasterStemmer()

    # Stem the words.
    return [stemmer.stem(t) for t in text]
