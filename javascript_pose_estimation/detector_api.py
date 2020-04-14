import tornado.ioloop
import tornado.web
import tornado.websocket
import subprocess
import os


import json
from time import sleep
import time
import argparse
import numpy as np
import cv2

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("dist/index.html")


class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        print("WebSocket opened")

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        print(u"You said: " + message)

    def on_close(self):
        print("WebSocket closed")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r'/dist/(.*)', tornado.web.StaticFileHandler, {'path': 'dist'}),        
        (r"/pose", SimpleWebSocket)
    ])

class Detector:
    def __init__(self):
        print(os.path.dirname(os.path.realpath(__file__)))
        app = make_app()
        app.listen(7777)
        print("Start Server ...")
        tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    detector = Detector()