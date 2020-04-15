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

from http.server import BaseHTTPRequestHandler, HTTPServer

def simple_post_server(bind_address, port, handler):

    class EndpointHttpServer(BaseHTTPRequestHandler):
        def do_POST(self):
            handler(self)

    webServer = HTTPServer((bind_address, port), EndpointHttpServer)
    print('Running webserver: %s:%s' % (bind_address, port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print('Server completed...')
