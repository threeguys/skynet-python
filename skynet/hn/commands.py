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

import argparse

import hn.data
import hn.api

class Command:
    def __init__(self, key, operation):
        self.key = key
        self.operation = operation

    def run(self, args):
        return self.operation(args)

class SeenCommand(Command):
    def __init__(self):
        super().__init__('gen-hn-seen', lambda args: hn.data.write_seen_file(args.db, args.seen))

class CrawlCommand(Command):
    def __init__(self):
        super().__init__('crawl-hn', lambda args: hn.api.crawl_missing(args.seen))

_all_commands = {
    'seen': SeenCommand,
    'crawl': CrawlCommand
}

def hn_cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--db', default='total.db', action='store', help='file containing existing records which have been read')
    parser.add_argument('-s', '--seen', default='hn-seen.db', action='store', help='file containing the seen ids in the database (for crawling)')
    parser.add_argument('action', default='walk', action='store', help='action to perform (walk|summary|missing|gaps)')
    args = parser.parse_args()

    constructor = _all_commands.get(args.action.lower())
    if constructor is not None:
        cmd = constructor()
        cmd.run(args)
