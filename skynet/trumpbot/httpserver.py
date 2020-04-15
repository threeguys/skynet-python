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

from skynet.endpoints.ephttp import simple_post_server
from skynet.seq2seq import Generator

def run_server(model_name, port=8080):
    generator = Generator(model_name)
    print('Test is: ' + generator.generate(500, 'test string'))

    def handler(server):
        try:
            if 'Content-Length' not in server.headers:
                server.send_error(400, 'Bad Request: Content-Length required')
                return

            if 'Content-Type' not in server.headers or server.headers['Content-Type'] != 'text/plain':
                server.send_error(400, 'Bad Request: Content-Type must be text/plain')

            input_len = int(server.headers['Content-Length'])
            input_text = server.rfile.read(input_len).decode('utf-8')
            if not input_text.endswith(('.','?','!')):
                input_text = input_text + '.'

            output_text = generator.generate(500, input_text)
            server.send_header('Content-Type', 'text/plain')
            server.send_header('Content-Length', str(len(output_text)))
            server.wfile.write(output_text.encode('utf-8'))

        except Exception as e:
            server.send_error(500, 'Internal Server Error')
            print(e)

    simple_post_server('0.0.0.0', port, handler)
