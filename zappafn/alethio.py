#!/usr/bin/python

import logging
import os
import sys

import boto3
import numpy as np
import pandas as pd

from stat_parser import Parser
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)
s3 = boto3.resource('s3')

# RDS was too expensive so download a sqlite3 db from s3
# note that lambda functions sometimes share containers?
# see https://forums.aws.amazon.com/thread.jspa?messageID=663677
if not os.path.exists('/tmp/alethio.sqlite'):
    logger.info('downloading sqlite db from s3')
    s3.Bucket('zappa-uploads-owrhowohry').download_file('alethio.sqlite', '/tmp/alethio.sqlite')
else:
    logger.info('sqlite db already downloaded from s3 on this container')

# connection for local sqlite database
ENGINE = create_engine('sqlite:////tmp/alethio.sqlite')
logger.info('success connecting to database: %s', ENGINE)

# each of these dataframes is indexed with unicode emoji characters.
# we load them into memory at the module level because that way they're shared
# across invocations of the lambda entry-point function and we don't have to
# reload them every time
logger.info('loading answer emoji from database')
EMOJI_ANSR = pd.read_sql('SELECT * FROM answers', ENGINE, index_col='unicode')

logger.info('loading emoji w2v from database')
EMOJI_VECS = (pd.read_sql('SELECT * FROM emoji_w2v', ENGINE, index_col='unicode')
                .apply(pd.to_numeric))
logger.info('success loading emoji dataframes')


class UnknownWord(ValueError):
    """Raised when we don't know the word2vec vector for a word"""
    pass


def _softmax(x):
    """Compute the softmax of the input"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_word_vec(word, conn=ENGINE):
    """Load the word2vec vector for the given word from the database

    Parameters
    ----------
    word: str
        A word to find word2vec representation for.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the word2vec representation of the given word

    Raises
    ------
    UnknownWord
        If the given word doesn't have a known word2vec representation
    """

    # first try exact word match (will be fast bc of index on 'word' column)
    query = "SELECT * FROM words_w2v WHERE word = '{}'".format(word)
    vector_df = pd.read_sql(query, conn, index_col='word')

    if not vector_df.empty:
        logger.debug(
            f'found exact match for query {word} in w2v')
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


def get_w2v_emoji_dist(word_vec, n=5):
    """Get the probability distribution over emoji for a word's word2vec
    representation.

    Probability values are the softmax'd cosine similarities between the word's
    word2vec representation and the word2vec representation of each emoji.

    Parameters
    ----------
    word_vec: pd.DataFrame
        A dataframe containing word2vec values for a word to be associated with
        an emoji.

    Returns
    -------
    pd.Series
        A series, indexed by emoji unicode values, containing a probability
        distribution over emoji related to the input word2vec vector.
    """

    # take the dot product of each emoji's w2v vector with the vector for
    # the given word, then convert to a pd.Series (the .iloc business) for
    # sorting
    cos_sim = EMOJI_VECS.dot(word_vec.transpose()).iloc[:, 0]

    top_n = cos_sim.sort_values(ascending=False).head(n)

    # convert the cosine similarities into a probability distribution
    top_n = _softmax(top_n)

    return pd.Series(data=top_n, index=top_n.index, name='prob')


def sample_from_dist(dist):
    """Randomly select an emoji according a given probability distribution.

    Parameters
    ----------
    dist : pd.Series
        Series of probabilities indexed by emoji unicode values

    Returns
    -------
    str
        Unicode value for the an emoji sampled according to the input
        probability distribution.
    """

    return np.random.choice(dist.index.values, p=dist.values)


def get_emoji(word):
    """Try to get an emoji relevant to the supplied word

    Parameters
    ----------
    word: str
        A word to retrieve an emoji for

    Returns
    -------
    str
        Unicode value for the selected emoji

    Raises
    ------
    UnknownWord
        If the given word can't be associated with an emoji
    """

    return sample_from_dist(get_w2v_emoji_dist(get_word_vec(word)))


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
