import os

import pyautogui
from pynput.keyboard import Key, Controller
import win32com.client


class PresentationWrapper:
    def __init__(self, presentation):
        self.presentation = presentation
        self.zoom = False
        self.keyboard = Controller()

    def run_slideshow(self):
        self.presentation.SlideShowSettings.Run()

    def close_slideshow(self):
        self.presentation.SlideShowWindow.View.Exit()

    def next_slide(self):
        self.presentation.SlideShowWindow.View.Next()

    def previous_slide(self):
        self.presentation.SlideShowWindow.View.Previous()

    def start_zoom(self):
        # self.presentation.Window.View.Zoom = 200

        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press('+')
            self.keyboard.release('+')
        self.keyboard.release(Key.cmd)

    def stop_zoom(self):
        # self.presentation.Window.View.Zoom = 30

        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press('-')
            self.keyboard.release('-')

class PowerpointWrapper:
    def __init__(self):
        self.application = None

    def start_powerpoint(self):
        if self.application is None:
            self.application = win32com.client.Dispatch("PowerPoint.Application")
            self.application.Visible = 1

    def open_presentation(self, filename) -> PresentationWrapper:
        if self.application is None:
            self.start_powerpoint()

        presentation = self.application.Presentations.Open(os.path.abspath(filename))
        return PresentationWrapper(presentation)

    def quit(self):
        self.application.Quit()