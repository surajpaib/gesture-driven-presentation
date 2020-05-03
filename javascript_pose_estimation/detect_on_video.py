import tornado.ioloop
import tornado.web
import tornado.websocket
import webbrowser
import subprocess
import json, time
import asyncio
import pyppeteer
from pyppeteer import launch

pyppeteer.DEBUG = True

def launch_browser():

    async def main():
        browser = await launch(
            headless=True,
            ignoreHTTPSErrors = True,
            args=[
                '--use-fake-device-for-media-stream',
                '--no-sandbox',
                '--use-file-for-fake-video-capture=/Users/admin/Desktop/test.y4m',
                '--disable-infobars',
                '--disable-web-security',
                '--use-fake-ui-for-media-stream',
                '--disable-infobars',
                '--no-sandbox',
                '--disable-web-security',
                '--ignore-certificate-errors',
                '--allow-file-access',
                '--unsafely-treat-insecure-origin-as-secure',
                '--start-maximized'
            ],
            # executablePath= "/usr/local/bin/chr0me"
        )
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
    ws_interface = WebSocketServer(port=port)
    time.sleep(1)
    launch_browser()

    # subprocess.Popen(["/usr/local/bin/chrome",
    #                   "--headless", "--disable-gpu", "http://localhost:{}/".format(port),
    #                   "--use-file-for-fake-video-capture={}".format(file)],
    #                  cwd=os.path.dirname(os.path.realpath(__file__)))

    try:
        ws_interface.start()
    finally:
        ws_interface.stop()