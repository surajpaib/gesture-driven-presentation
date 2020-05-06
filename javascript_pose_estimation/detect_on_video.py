import tornado.ioloop
import tornado.web
import tornado.websocket
import webbrowser
import subprocess
import json, time
import asyncio
import pickle


buffer = []
c = [0]


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        print('GOT')
        self.render("dist/index.html")

class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        print("WebSocket opened")

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        response = json.loads(message)
        if response == "VIDEO END":
            c[0] += 1
            pickle.dump(buffer, open('buffer'+ str(c[0]) + '.pkl', 'wb'))
        else:
            buffer.append(response)
        # print(response_dict)

    def on_close(self):
        print("WebSocket closed")


def get_tornado_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r'/dist/(.*)', tornado.web.StaticFileHandler, {'path': 'dist'}),
        (r'/videos/(.*)', tornado.web.StaticFileHandler, {'path': 'videos'}),
        (r"/pose", SimpleWebSocket)
    ])


class WebSocketServer:
    def __init__(self, port):
        self.port = port
        self.app = get_tornado_app()
        self.app.listen(self.port)

    def start(self):
        # webbrowser.open_new("http://localhost:{}".format(self.port))
        tornado.ioloop.IOLoop.current().start()

    def stop(self):
        print("On close handler!")
        tornado.ioloop.IOLoop.current().stop()

if __name__ == "__main__":
    port = 7777
    ws_interface = WebSocketServer(port=port)
    time.sleep(1)
    # launch_browser()

    # subprocess.Popen(["/usr/local/bin/chrome",
    #                   "--headless", "--disable-gpu", "http://localhost:{}/".format(port),
    #                   "--use-file-for-fake-video-capture={}".format(file)],
    #                  cwd=os.path.dirname(os.path.realpath(__file__)))

    try:
        ws_interface.start()
    finally:
        ws_interface.stop()