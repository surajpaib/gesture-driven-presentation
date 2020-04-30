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
        # Initialize Body Classification handler class
        self.body_classification_handler = BodyClassificationHandler()
        print("WebSocket opened")

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        """
        message: String received from javascript websocket
        This function is called everytime a message is received from the socket connection. 
        Message can be parsed using a json loads
        """
        response_dict = json.loads(message)
        print(response_dict)
        # Body classifcation handler update is called to send the body pose details to the handler.
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
        # webbrowser.open_new("http://localhost:{}".format(self.port))
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
