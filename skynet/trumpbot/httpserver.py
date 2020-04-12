
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
