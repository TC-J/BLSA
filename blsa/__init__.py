from types import NoneType
import cv2 as cv
import numpy as np
from .camera import Camera
from face_recognition import face_locations


def showfeed(cam):
    while True:
        frame = cam.capture(1)

        cv.imshow("frame", frame[0])

        if cv.waitKey(1) == ord("q"):
            cam.release()
            break


def main():
    cam = Camera(0)
    while True:
        frame = cam.capture(1)[0]
        detector = cv.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx", "", (320, 320)
        )
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        count, faces = detector.detect(frame)
        if count > 0 and not isinstance(faces, NoneType):
            for face in faces:
                coords = face[:-1].astype(np.int32)
                cv.rectangle(
                    frame,
                    (coords[0], coords[1]),
                    (coords[0] + coords[2], coords[1] + coords[3]),
                    (0, 255, 0),
                    1,
                )
                cv.circle(frame, (coords[4], coords[5]), 2, (0, 0, 255), 2)
                cv.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), 2)
                cv.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), 2)
                cv.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), 2)
                cv.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        cv.imshow("name", frame)

        # cv.rectangle(frame, (top, left), (bottom, right), (0, 255, 0), 3)

        if cv.waitKey(1) == ord("q"):
            cam.release()
            break


if __name__ == "__main__":
    main()
