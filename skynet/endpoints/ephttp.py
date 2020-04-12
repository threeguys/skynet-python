
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
