import tornado.ioloop
import tornado.web
import tornado.websocket
import webbrowser
import subprocess
import json, os
import asyncio
from pyppeteer import launch

def launch_browser():

    async def main():
        browser = await launch()
        page = await browser.newPage()
        await page.goto('http://localhost:7777')
        # await browser.close()

    asyncio.get_event_loop().run_until_complete(main())

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
        response_dict = json.loads(message)
        print(response_dict)

    def on_close(self):
        print("WebSocket closed")


def get_tornado_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r'/dist/(.*)', tornado.web.StaticFileHandler, {'path': 'dist'}),
        (r"/pose", SimpleWebSocket)
    ])


class WebSocketServer:
    def __init__(self, port):
        self.port = port
        self.app = get_tornado_app()
        self.app.listen(self.port)

    def start(self):
        tornado.ioloop.IOLoop.current().start()

    def stop(self):
        print("On close handler!")
        tornado.ioloop.IOLoop.current().stop()

if __name__ == "__main__":
    port = 7777
    file = "test.webm"
    ws_interface = WebSocketServer(port=port)

    # subprocess.Popen(["/usr/local/bin/chrome",
    #                   "--headless", "--disable-gpu", "http://localhost:{}/".format(port),
    #                   "--use-file-for-fake-video-capture={}".format(file)],
    #                  cwd=os.path.dirname(os.path.realpath(__file__)))

    try:
        ws_interface.start()
    finally:
        ws_interface.stop()