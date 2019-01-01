#!/usr/bin/env python

import datetime as dt
import os
import logging

import awslogging
import boto3
import twython
import alethio
import tweet_utils


s3 = boto3.resource('s3')

logger = logging.getLogger(__name__)


def main(json_input, context):

    logger.info('getting twython client')
    client = twython.Twython(
        os.getenv("twitter_consumer_key"),
        os.getenv("twitter_consumer_secret"),
        os.getenv("twitter_access_token_key"),
        os.getenv("twitter_access_token_secret"))

    logger.info('reading latest processed from s3')
    latest_processed = read_latest_processed(s3)
    logger.info('latest: %s', latest_processed)

    logger.info('getting latest tweets')
    new_tweets = \
        client.get_mentions_timeline(since_id=int(latest_processed))

    logger.info('processing {} tweets'.format(len(new_tweets)))
    for tweet in new_tweets:
        latest_processed = process_tweet(client, tweet)
        record_latest_processed(latest_processed, s3)
        logger.info('processed {}'.format(latest_processed))


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

    status_message = "@" + user + " " + answer

    client.update_status(status=status_message, in_reply_to_status_id=id_str)
    logger.info(status_message)


def read_latest_processed(s3):
    """Read filenames from s3 bucket as hacky db entries"""

    # TODO: make this configurable
    bucket = s3.Bucket('alethiomoji-processed-ids')

    latest_processed = max([obj.key for obj in bucket.objects.all()])

    return latest_processed


def record_latest_processed(latest_processed, s3):
    """Record filenames in s3 bucket as hacky db entries"""

    filename = f'/tmp/{latest_processed}'

    with open(filename, 'w') as outfile:
        outfile.write(' ')

    s3.meta.client.upload_file(filename, 'alethiomoji-processed-ids', latest_processed)


if __name__ == '__main__':
    main(None, None)
