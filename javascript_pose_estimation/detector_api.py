import tornado.ioloop
import tornado.web
import tornado.websocket
import webbrowser
import os


import json
from time import sleep
import time
import argparse
import numpy as np
import cv2
from classification_handler import BodyClassificationHandler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("dist/index.html")


class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        self.body_classification_handler = BodyClassificationHandler()
        print("WebSocket opened")

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        response_dict = json.loads(message)
        self.body_classification_handler.update(response_dict["body_pose"][0], xmax=response_dict["image_width"], ymax=response_dict["image_height"])

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
        self.port = 7777
        self.app = get_tornado_app()
        self.app.listen(self.port)

    def start(self):
        webbrowser.open_new("http://localhost:{}".format(self.port))
        tornado.ioloop.IOLoop.current().start()

    def stop(self):
        print("On close handler!")
        tornado.ioloop.IOLoop.current().stop()



if __name__ == "__main__":
    ws_interface = WebSocketServer(port=7777)

    try:
        ws_interface.start()
    finally:
        ws_interface.stop()
