#  Copyright 2020 Ray Cole
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import praw
import os
import time

from skynet.seq2seq import Generator

def run_reddit_bot(model_name, profile_name, user_agent):
    generator = Generator(model_name)
    replied = set([])

    while True:
        try:
            reddit = praw.Reddit(profile_name, user_agent=user_agent)
            group = reddit.subreddit('SkynetProvingGrounds')

            while True:
                # Get new posts
                for submission in group.new():
                    if submission.id not in replied:
                        input_text = submission.selftext
                        if input_text is None or len(input_text) == 0:
                            input_text = submission.title
                        if input_text is None or len(input_text) == 0:
                            input_text = submission.name

                        if input_text is None or len(input_text) == 0:
                            print("Could not deduce input text!")
                            continue
                        
                        if not input_text.endswith(('.','?','!')):
                            input_text = input_text + '.'

                        found = False
                        for comment in submission.comments.list():
                            if comment.author.name == profile_name:
                                found = True
                                print('Found reply for %s' % submission.id)
                                replied.add(submission.id)
                                break

                        if not found:
                            # generate reply and make a test reply
                            print('INPUT[%s]: %s' % (submission.id, input_text))
                            output_text = generator.generate(500, input_text)
                            print('REPLY[%s]: %s' % (submission.id, output_text))
                            submission.reply(output_text)
                            replied.add(submission.id)

                time.sleep(30)

        except Exception as e:
            print('Error in comment loop, reconnecting in 5 minutes...', e)

        time.sleep(300)

