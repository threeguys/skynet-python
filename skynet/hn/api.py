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

import http.client
import json
import time

import skylogging
import hn.data

class Client:
    def __init__(self):
        self.conn = http.client.HTTPSConnection('hacker-news.firebaseio.com')

    def execute(self, url):
        self.conn.request('GET', url)
        resp = self.conn.getresponse()
        return json.loads(resp.read())

    def get_item(self, index):
        return self.execute('/v0/item/%s.json' % index)

    def get_max_item(self):
        return self.execute('/v0/maxitem.json')


class Crawler:
    def __init__(self, db=None, log=None):
        self.client = Client()
        self.seen = {}
        self.pending = []
        self.log = log if log is not None else skylogging.Log()

    def seen_minmax(self):
        min_id = None
        max_id = None
        for id in self.seen.keys():
            if min_id is None or id < min_id:
                min_id = id
            if max_id is None or id > max_id:
                max_id = id
        return (min_id, max_id)

    def mark_pending(self, id):
        def mark_one(one):
            if one not in self.seen:
                self.pending.append(one)

        if id is not None:
            if type(id) is list:
                [mark_one(i) for i in id]
            else:
                mark_one(id)

    def mark_seen(self, id):
        def mark_one(x):
            self.seen[x] = True

        if id is not None:
            if type(id) is list:
                [mark_one(x) for x in id]
            else:
                mark_one(id)

    def next_missing(self, start_id):
        current = start_id
        while current > 1:
            if current not in self.seen:
                return current
            current -= 1
        return 0

    def walk(self, start_id=None):
        start = start_id if start_id is not None else self.client.get_max_item()
        self.pending.append(start)

        while len(self.pending) > 0:
            target = self.pending.pop()
            if target not in self.seen:
                self.seen[target] = True
                item = self.client.get_item(target)
                if item is not None:
                    yield item

                    self.mark_pending(item.get('parent'))
                    self.mark_pending(item.get('kids'))
                else:
                    yield {
                        'id': target,
                        'dead': True,
                        'was_null': True
                    }

def crawl_missing(seen_path, log_obj=None):
    crawler = Crawler()
    log = log_obj if log_obj is not None else skylogging.Log()

    log.info('Loading seen ids from %s' % seen_path)

    start = time.time()

    # Load existing items
    with open(seen_path, 'rt') as seen_ids:
        for line in seen_ids:
            data = line.rstrip()
            if len(data) > 0:
                crawler.mark_seen(int(data))

    log.info('Loaded %d items in %d seconds' % (len(crawler.seen), time.time() - start))

    done = False
    counter = 0
    min_id, seed_id = crawler.seen_minmax()
    while not done:
        for item in crawler.walk():
            if item is not None:
                counter += 1
                log.data(json.dumps(item))

        if seed_id > 1:
            for i in range(0, 5):
                seed_id = crawler.next_missing(seed_id)
                crawler.mark_pending(seed_id)
        else:
            done = True
        log.info('Total documents: %d' % counter)
