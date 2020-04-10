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

import json
import html

import skylogging

class Filters:

    def _filter(self, field, value):
        return lambda x: field in x and x[field] == value

    def by(self, name):
        return self._filter('by', name)

    def deleted(self):
        return self._filter('deleted', True)

    def dead(self):
        return self._filter('dead', True)

    def rectype(self, typename):
        return self._filter('type', typename)

    def comment(self, user=None):
        if user is None:
            return self.rectype('comment')
        else:
            return self.ands([ self.rectype('comment'), self.by(user) ])

    def del_or_dead(self):
        return self.ors([ self.deleted(), self.dead() ])

    def live(self):
        return self.complement(self.del_or_dead())

    def ors(self, filters):
        def or_filter(item):
            for f in filters:
                if f(item):
                    return True
            return False
        return or_filter

    def ands(self, filters):
        def and_filter(item):
            for f in filters:
                if not f(item):
                    return False
            return True
        return and_filter

    def complement(self, filter):
        return lambda x: not filter(x)

class Reader:
    def __init__(self, path, log=None):
        self.path = path
        self.filters = Filters()
        self.log = log if log is not None else skylogging.Log()

    def records(self, item_filter=None):
        with open(self.path, 'r') as f:
            line_no = 0
            for line in f:
                line_no += 1
                try:
                    obj = json.loads(line.rstrip())
                    if obj is not None and (item_filter is None or item_filter(obj)):
                        yield obj
                except Exception as e:
                    self.log.error('Error at line %d: "%s"' % (line_no, line))

    def comments(self, item_filter=None):
        comment_filter = self.filters.ands([ self.filters.comment(), self.filters.live(), lambda x: item_filter is None or item_filter(x) ])
        for i in self.records(item_filter=comment_filter):
            yield i

    def comments_by(self, user):
        return self.comments(self.filters.by(user))

def write_seen_file(input, output, log_obj=None):
    reader = Reader(input, log_obj)
    with open(output, 'wt') as outfile:
        for r in reader.records():
            print('%d' % r['id'], file=outfile)
