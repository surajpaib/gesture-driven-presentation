import tornado.ioloop
import tornado.web
import tornado.websocket
import webbrowser
import subprocess
import json, time
import asyncio
import pyppeteer
from pyppeteer import launch

pyppeteer.DEBUG = False



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


    def launch_browser(self):
        async def main():
            browser = await launch(
                headless=False,
                ignoreHTTPSErrors = True,
                args=[

                    '--use-fake-device-for-media-stream',
                    '--use-fake-ui-for-media-stream',
                    '--use-file-for-fake-video-capture=./test_gesture.y4m',
                    '--no-sandbox',
                    '--disable-infobars',
                    '--disable-web-security',
                    '--ignore-certificate-errors',
                    '--allow-file-access',
                    '--unsafely-treat-insecure-origin-as-secure',
                    '--enable-webgl',
                    '--hide-scrollbars',
                    '--mute-audio',
                    '--no-first-run',
                    '--disable-infobars',
                    '--disable-breakpad',
                ],
                executablePath= "/usr/bin/google-chrome"
            )
            page = await browser.newPage()
            await page.goto('http://localhost:7777')
            await page.waitForSelector('#main', visible=True)
            await page.waitFor(3000)
            await browser.close()
            self.stop()
            

        asyncio.get_event_loop().run_until_complete(main())


    def start(self):
        self.launch_browser()
        tornado.ioloop.IOLoop.current().start()

    def stop(self):
        print("On close handler!")
        tornado.ioloop.IOLoop.current().stop()

if __name__ == "__main__":
    port = 7777
    ws_interface = WebSocketServer(port=port)
    time.sleep(1)
    ws_interface.start()
