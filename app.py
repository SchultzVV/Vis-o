from scripts.detector import detector
from scripts.centroidtracker import CentroidTracker, TrackableObject
import numpy as np
import cvlib as cv
import datetime
import csv
import time
import dlib
import sys
import cv2
from flask_opencv_streamer.streamer import Streamer
from console_logging.console import Console
console = Console()


#port = 9092
port = 8081
require_login = False
streamer = Streamer(port, require_login)

#camera = "video/n-a50189cb4fa7a686,ch3_1657897377651[1].mp4"
camera = "video/a.mp4"


def find_bbox(rects, centroid):
    for (i, (startX, startY, endX, endY)) in enumerate(rects):
        cX = int((startX + endX) / 2)
        cY = int((startY + endY) / 2)
        if (cX == centroid[0]) and (cY == centroid[1]):
            return startX, startY, endX, endY


def main():

    dict_detect = dict()
    file_csv = open('detection.csv', 'w')
    fieldnames = ['ID', 'Direcao', 'Horario', 'Data', 'Genero']
    writer = csv.DictWriter(file_csv, fieldnames=fieldnames)
    writer.writeheader()

    prev_frame_time = 0
    new_frame_time = 0
    padding = 20

    roi_ = (150,530,520,680)
    line_i = (0, 120)
    line_f = (int(abs(roi_[1]-roi_[3])), 120)

    ct = CentroidTracker(maxDisappeared=30, maxDistance=30)
    trackers = []
    trackableObjects = {}

    frames = 0
    entrada = 0
    saida = 0

    video_capture = cv2.VideoCapture(camera)

    while True:
        ok, frame = video_capture.read()
        if ok:
        
            #roi = frame[150:480, 480:730]
            roi = frame[roi_[0]:roi_[2],roi_[1]:roi_[3]]

            new_frame_time = time.time()
            rects = []

            detect = detector(roi)

            if frames % 20 == 0:
                trackers = []

                for (startX, startY, endX, endY) in detect.detection_filter():
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)

                    tracker.start_track(roi, rect)
                    trackers.append(tracker)
            else:
                for tracker in trackers:
                    tracker.update(roi)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    if startY < 0:
                        startY = 0
                    if startX < 0:
                        startX = 0
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    cv2.putText(roi, 'person', (startX-5, startY-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.rectangle(roi, (startX, startY),
                       (endX, endY), (255, 255, 255), 2)
                    rects.append((startX, startY, endX, endY))

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)
                cv2.circle(roi, centroid, 2, color=(
                    255, 255, 255), thickness=-1)

                if to is None:
                    to = TrackableObject(objectID, centroid)

                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < 0 and detect.detect_in_line(centroid, line_i, line_f):
                            label = 0
                            bbox = find_bbox(rects, centroid)
                            img = roi[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            #cv2.imshow('image window', img)
                            face, confidence = cv.detect_face(img)
                            for idx, f in enumerate(face):
                                try:
                                    (startX, startY) = max(
                                        0, f[0]-padding), max(0, f[1]-padding)
                                    (endX, endY) = min(
                                        img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
                                    face_crop = np.copy(
                                        img[startY:endY, startX:endX])
                                    cv2.imshow('image window', face_crop)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    (label, confidence) = cv.detect_gender(
                                        face_crop)
                                    idx = np.argmax(confidence)
                                    label = label[idx]
                                    # label = "{}: {:.2f}%".format(label, confidence[idx] * 100)# Uncoment for probabilities
                                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                                    cv2.putText(
                                        img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                except:
                                    label = 0
                            saida += 1
                            to.counted = True
                            dict_detect.update({'ID': objectID, 'Direcao': 'Saida',
                                                'Horario': datetime.datetime.now().strftime("%H:%M:%S"), 'Data': datetime.date.today(), 'Genero': label})
                        elif direction > 0 and detect.detect_in_line(centroid, line_i, line_f):
                            label = 0
                            bbox = find_bbox(rects, centroid)
                            print(bbox)
                            img = roi[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            face, confidence = cv.detect_face(img)
                            for idx, f in enumerate(face):
                                try:
                                    (startX, startY) = max(
                                        0, f[0]-padding), max(0, f[1]-padding)
                                    (endX, endY) = min(
                                        img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
                                    face_crop = np.copy(
                                        img[startY:endY, startX:endX])
                                    cv2.imshow('image window', face_crop)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    (label, confidence) = cv.detect_gender(
                                        face_crop)
                                    idx = np.argmax(confidence)
                                    label = label[idx]
                                    # label = "{}: {:.2f}%".format(label, confidence[idx] * 100)# Uncoment for probabilities
                                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                                    cv2.putText(
                                        img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    print(label)
                                except:
                                    label = 0

                            entrada += 1
                            to.counted = True
                            dict_detect.update({'ID': objectID, 'Direcao': 'Entrada',
                                                'Horario': datetime.datetime.now().strftime("%H:%M:%S"), 'Data': datetime.date.today(), 'Genero': label})

                trackableObjects[objectID] = to

            if len(dict_detect) > 0:
                writer.writerow(dict_detect)
                console.info(dict_detect)
            dict_detect.clear()

            cv2.rectangle(frame, (roi_[1], roi_[0]), (roi_[3], roi_[2]), (255, 255, 255), 2)
            cv2.line(roi, line_i, line_f, (0, 255, 0), 2)

            cv2.putText(roi, 'S: {}'.format(saida), (5, 210),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(roi, 'E: {}'.format(entrada), (5, 225),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

            streamer.update_frame(frame)
            if not streamer.is_streaming:
                streamer.start_streaming()

            frames += 1

        else:
            break


if __name__ == "__main__":
    main()
