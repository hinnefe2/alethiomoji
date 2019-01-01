#!/usr/bin/python

import logging
import os
import sys

import numpy as np
import pandas as pd

from stat_parser import Parser
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)

# connection for AWS RDS postgres database
logger.info('trying to connect to postgres at %s', os.getenv('PGDATABASE'))
ENGINE = create_engine(
        'postgresql://{user}:{password}@{host}/{dbname}'.format(
            dbname=os.getenv('PGDATABASE'),
            host=os.getenv('PGHOST'),
            password=os.getenv('PGPASSWORD'),
            user=os.getenv('PGUSER')))

logger.info('success connecting to postgres: %s', ENGINE)

# each of these dataframes is indexed with unicode emoji characters
logger.info('loading emoji dataframes from postgres')
EMOJI_ANNT = pd.read_sql('SELECT * FROM annotated', ENGINE, index_col='unicode')

logger.info('done with EMOJI_ANNT')
EMOJI_ANSR = pd.read_sql('SELECT * FROM answers', ENGINE, index_col='unicode')

logger.info('done with EMOJI_ANSR')
EMOJI_VECS = (pd.read_sql('SELECT * FROM emoji_w2v', ENGINE, index_col='unicode')
                .apply(pd.to_numeric))
logger.info('success loading emoji dataframes')


class UnknownWord(ValueError):
    """Raised when we don't know the word2vec vector for a word"""
    pass


def get_word_vec(word, conn=ENGINE):
    "Load the word2vec vector for the given word from the database"

    # first try exact word match (will be fast bc of index on 'word' column)
    query = "SELECT * FROM words_w2v WHERE word = '{}'".format(word)
    vector_df = pd.read_sql(query, conn, index_col='word')

    if not vector_df.empty:
        logger.debug('found exact match in w2v')
        return vector_df.astype(float)

    # next try LIKE match, will be slow (~20s)
    query = "SELECT * FROM words_w2v WHERE word LIKE '{}'".format(word.lower())
    vector_df = pd.read_sql(query, conn, index_col='word')

    if not vector_df.empty:
        # return the first one found, if multiple
        logger.debug('found LIKE  match in w2v')
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

    raise UnknownWord

    # vectorizer = CountVectorizer().fit(EMOJI_ANNT.annotation)
    # count_matrix = vectorizer.transform(EMOJI_ANNT.annotation)

    # transformer = TfidfTransformer()
    # count_tfidf = transformer.fit_transform(count_matrix)

    # check if the given word is included in any of the annotations
    if word not in vectorizer.vocabulary_:
        raise UnknownWord
    else:
        logger.debug('found match in annotations')

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
        logger.debug('not a question')
        return None

    # loop through the different types of words
    for tag_type in [noun_tags, adj_tags, verb_tags]:

        for word in get_words_by_tags(parse_tree, tag_type):
            try:
                logger.debug('trying to get emoji for {}'.format(word))

                emoji = get_emoji(word)
                answer.append(emoji)

                logger.debug('using {} for {}'.format(emoji, word))

            except UnknownWord:
                pass

    # add an emoji that indicates some answer (yes/no/maybe/good/bad/etc)
    logger.debug('adding an answer emoji')
    answer.append(get_answer_emoji())

    # shuffle the order of the emojis
    logger.debug('shuffling the order')
    np.random.shuffle(answer)

    logger.debug('returning final response:')
    return ' '.join(answer)


if __name__ == '__main__':

    # if run from the command line assume everything else on the line is the
    # question
    question = ' '.join(sys.argv[1:])
    print(generate_answer(question))
