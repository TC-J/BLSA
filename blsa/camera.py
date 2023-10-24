import cv2 as cv
import numpy


class Camera:
    def __init__(self, nr):
        self.camera = cv.VideoCapture(nr)

    def capture(self, count):
        frames = list()
        while count > 0:
            _, frame = self.camera.read()
            frames.append(frame)
            count -= 1
        return frames

    def release(self):
        self.camera.release()
