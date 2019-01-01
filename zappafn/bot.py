#!/usr/bin/env python

import datetime as dt
import os
import logging

import twython
import alethio
import tweet_utils


logger = logging.getLogger(__name__)


def main(json_input, context):

    logger.info('getting twython client')
    client = twython.Twython(
        os.getenv("twitter_consumer_key"),
        os.getenv("twitter_consumer_secret"),
        os.getenv("twitter_access_token_key"),
        os.getenv("twitter_access_token_secret"))

    logger.info('reading latest processed from postgres')
    latest_processed = read_latest_processed()
    logger.info('latest: %s', latest_processed)

    logger.info('getting latest tweets')
    new_tweets = \
        client.get_mentions_timeline(since_id=int(latest_processed))

    logger.info('processing {} tweets'.format(len(new_tweets)))
    for tweet in new_tweets:
        latest_processed = process_tweet(client, tweet)
        record_processing(latest_processed)


def process_tweet(client, tweet):

    logger.debug('in process_tweet')

    id_str = tweet['id_str']
    user = tweet['user']['screen_name']
    text = tweet_utils.get_text_cleaned(tweet)

    logger.info(text)

    answer = alethio.generate_answer(text)

    if answer is not None:
        logger.debug(' ... got answer')
        send_response(client, user, id_str, answer)
    else:
        answer = "Sorry, I couldn't understand your question"
        logger.debug(' ... got None for answer')
        send_response(client, user, id_str, answer)

    return id_str


def send_response(client, user, id_str, answer):

    logger.debug('in send_response')

    status_message = ".@" + user + " " + answer

    client.update_status(status=status_message, in_reply_to_status_id=id_str)
    logger.info(status_message)


def read_latest_processed(engine=alethio.ENGINE):

    sql = "SELECT tweet_id FROM processed ORDER BY processed_at DESC LIMIT 1"

    with engine.connect() as conn:
        latest_processed = conn.execute(sql).fetchone()[0]

    return latest_processed


def record_processing(latest_processed, engine=alethio.ENGINE):

    sql = "INSERT INTO processed(processed_at, tweet_id)  VALUES (%s, %s)"

    with engine.connect() as conn:
        conn.execute(sql, (dt.datetime.now(), latest_processed))


if __name__ == '__main__':
    main(None, None)
