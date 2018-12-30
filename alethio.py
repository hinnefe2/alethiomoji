#!/usr/bin/python

import logging
import os
import sys

import numpy as np
import pandas as pd
import psycopg2 as pg2

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from stat_parser import Parser

LOGGER = logging.getLogger(__name__)

# connection for AWS RDS postgres database
CONN = pg2.connect(dbname=os.getenv('PGDATABASE'),
                   host=os.getenv('PGHOST'),
                   password=os.getenv('PGPASSWORD'),
                   user=os.getenv('PGUSER'))

# each of these dataframes is indexed with unicode emoji characters
EMOJI_ANNT = pd.read_sql('SELECT * FROM annotated', CONN, index_col='unicode')
EMOJI_ANSR = pd.read_sql('SELECT * FROM answers', CONN, index_col='unicode')
EMOJI_VECS = (pd.read_sql('SELECT * FROM emoji_w2v', CONN, index_col='unicode')
                .apply(pd.to_numeric))


class UnknownWord(ValueError):
    """Raised when we don't know the word2vec vector for a word"""
    pass


def _build_annotated_table(annot_file="annotations.csv"):
    """Build the sqlite table 'annotated' using the provided file"""

    annot_df = pd.read_csv(annot_file)
    annot_df.to_sql('annotated', CONN, if_exists='replace')


def _build_answer_table(answer_file="answers.csv"):
    "Build the sqlite table 'answers' using the provided file"""

    answer_df = pd.read_csv(answer_file)
    answer_df.to_sql('answers', CONN, if_exists='replace')


def get_word_vec(word, conn=CONN):
    "Load the word2vec vector for the given word from the database"

    # first try exact word match (will be fast bc of index on 'word' column)
    query = "SELECT * FROM words_w2v WHERE word = '{}'".format(word)
    vector_df = pd.read_sql(query, conn, index_col='word')

    if not vector_df.empty:
        LOGGER.debug('found exact match in w2v')
        return vector_df.astype(float)

    # next try LIKE match, will be slow (~20s)
    query = "SELECT * FROM words_w2v WHERE word LIKE '{}'".format(word.lower())
    vector_df = pd.read_sql(query, conn, index_col='word')

    if not vector_df.empty:
        # return the first one found, if multiple
        LOGGER.debug('found LIKE  match in w2v')
        return vector_df.head(1).astype(float)

    # finally, if nothing found raise an error
    raise UnknownWord


def get_w2v_emoji_dist(word, n=3):
    "Get the probability distribution over emoji for `word` using word2vec."

    try:
        word_vec = get_word_vec(word)

        # take the dot product of each emoji's w2v vector with the vector for
        # the given word, then convert to a pd.Series (the .iloc business) for
        # sorting
        cos_sim = EMOJI_VECS.dot(word_vec.transpose()).iloc[:, 0]

        top_n = cos_sim.sort_values(ascending=False).head(n)

        # shift up so that all values are >= 0, then normalize
        top_n = (top_n + min(top_n))**2
        top_n = top_n / sum(top_n)

        return pd.Series(data=top_n, index=top_n.index, name='prob')

    except UnknownWord:

        # return an empty series if we don't have word2vec vectors for the
        # input word
        return pd.Series(name='prob')


def get_annot_emoji_dist(word):  # tfidf=count_tfidf, vectorizer=vectorizer):
    "Get the probability distribution over emoji for `word` using annotations"

    vectorizer = CountVectorizer().fit(EMOJI_ANNT.annotation)
    count_matrix = vectorizer.transform(EMOJI_ANNT.annotation)

    transformer = TfidfTransformer()
    count_tfidf = transformer.fit_transform(count_matrix)

    # check if the given word is included in any of the annotations
    if word not in vectorizer.vocabulary_:
        raise UnknownWord
    else:
        LOGGER.debug('found match in annotations')

    # word must be in a list, otherwise vectorizer splits it into characters
    # instead of words
    word_vector = vectorizer.transform([word])

    # calculate cosine similarity of the word vector with each emoji's
    # annotation vector
    cos_sim = count_tfidf.dot(word_vector.transpose())

    # reshape to 1d array
    cos_sim = cos_sim.toarray().reshape(count_tfidf.shape[0],)

    # normalization, assumes all cosine similarities are positive
    if sum(cos_sim) != 0:
        norm_dist = cos_sim / sum(cos_sim)
    else:
        norm_dist = cos_sim

    return pd.Series(data=norm_dist, index=EMOJI_ANNT.index, name='prob')


def sample_from_dist(dist):
    """Randomly select an emoji according a given probability distribution.

    Parameters
    ----------
    dist : pd.Series
        Series of probabilities indexed by emoji unicode values

    Returns
    -------
    str
        Unicode value for the selected emoji
    """

    if any(dist):
        # assumes the dist is a pd.Series of probabilities indexed by the
        # unicode for each emoji
        return np.random.choice(dist.index.values, p=dist.values)
    else:
        raise UnknownWord


def get_emoji(word):
    """Try to get an emoji relevant to the supplied word"""

    # first try to sample from the annotated data if possible
    try:
        emoji_dist = get_annot_emoji_dist(word)
        return sample_from_dist(emoji_dist)
    except UnknownWord:
        pass

    # next try to sample from the word2vec data if possible
    try:
        emoji_dist = get_w2v_emoji_dist(word)
        return sample_from_dist(emoji_dist)
    except UnknownWord:
        pass

    # finally raise an exception if neither option worked
    raise UnknownWord


def get_answer_emoji(emoji_answer=EMOJI_ANSR):
    """Randomly select an emoji from a set which can be interpreted as yes/no/maybe

    Parameters
    ----------
    emoji_answer : pd.DataFrame
        A dataframe containing emoji which can be interpreted as yes/no/maybe.
        The dataframe is indexed by the emoji unicode values.

    Returns
    -------
    str
        An emoji character which can be interpreted as a yes/no/maybe
    """

    return np.random.choice(emoji_answer.index)


def generate_answer(question):

    def get_words_by_tags(parse_tree, tags):
        return [word[0] for word in parse_tree.pos() if word[1] in tags]

    noun_tags = ['NN', 'N', 'NNS']
    adj_tags = ['JJ']
    verb_tags = ['VB']

    answer = []

    parse_tree = Parser().parse(question)

    # if the provided question isn't parsed as a question
    if 'Q' not in parse_tree.label():
        LOGGER.debug('not a question')
        return None

    # loop through the different types of words
    for tag_type in [noun_tags, adj_tags, verb_tags]:

        for word in get_words_by_tags(parse_tree, tag_type):
            try:
                LOGGER.debug('trying to get emoji for {}'.format(word))
                answer.append(get_emoji(word))
            except UnknownWord:
                pass

    # add an emoji that indicates some answer (yes/no/maybe/good/bad/etc)
    LOGGER.debug('adding an answer emoji')
    answer.append(get_answer_emoji())

    # shuffle the order of the emojis
    LOGGER.debug('shuffling the order')
    np.random.shuffle(answer)

    LOGGER.debug('returning final response:')
    return ' '.join(answer)


if __name__ == '__main__':

    # if run from the command line assume everything else on the line is the
    # question
    question = ' '.join(sys.argv[1:])
    print(generate_answer(question))
