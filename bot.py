#!/usr/bin/env python

import json
import time
import logging

import twython
import alethio
import tweet_utils

# set up some simple logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def main():

    with open('credentials.json') as f:
            credentials = json.loads(f.read())

    client = twython.Twython(
        credentials["consumer_key"],
        credentials["consumer_secret"],
        credentials["access_token_key"],
        credentials["access_token_secret"])

    latest_processed = credentials['latest_processed']

    logger.info("Entering 'while True' loop")
    while True:

        try:
            new_tweets = \
                client.get_mentions_timeline(since_id=latest_processed)

            for tweet in new_tweets:
                latest_processed = process_tweet(client, tweet)

            save_progress(credentials)
            time.sleep(60)

        except KeyboardInterrupt:
            credentials['latest_processed'] = latest_processed
            save_progress(credentials)
            break

        except Exception as e:
            logging.exception(e)
            time.sleep(60)


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

    return id_str


def send_response(client, user, id_str, answer):

    logger.debug('in send_response')

    status_message = "@" + user + " " + answer

    client.update_status(status=status_message, in_reply_to_status_id=id_str)
    logger.info(status_message)


def save_progress(credentials):

    logger.debug('in save_progress')

    with open('credentials.json', 'w') as f:
            f.write(json.dumps(credentials))


if __name__ == '__main__':
    main()
