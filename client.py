import argparse
import sys
import numpy as np
import os
import sys
import time
from multiprocessing import Process, Queue
import threading
import re
import logging
import subprocess
import hashlib
import requests
import cv2 as cv
import json
from multiprocessing import Pool, TimeoutError

session = requests.Session()

def handle_response(fname, resp):
    try:
        response = resp.json()
    except:
        logging.exception("Invalid response {}".format(resp.text))
        return

    predictions = response
    cap = cv.VideoCapture(fname)
    has_frame, frame = cap.read()
    if not has_frame:
        return

    print("Geometry: {} x {}".format(frame.shape[1], frame.shape[0]))
    print(json.dumps(predictions, indent=4))

    pred_classes_dict = dict()

    for prediction in predictions:
        box = prediction['box']
        cv.rectangle(frame, (box['x'], box['y']), (box['x1'], box['y1']), (0, 0, 255), 2)

        label = "{}:{}".format(prediction['class'], prediction['confidence'])
        label_size, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(box['y'], label_size[1])

        cv.rectangle(frame, (box['x'], top - label_size[1]), (box['x'] + label_size[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (box['x'], top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        pred_classes_dict[prediction['class']] = prediction['confidence']

    classes_str = ""
    for k,v in enumerate(pred_classes_dict):
        classes_str += " {}:{}".format(k,v)

    if len(pred_classes_dict) > 0:
        h = hashlib.sha256()
        f = open(fname, "rb")
        h.update(f.read())
        f.close()

        outfile_path = re.sub("\.jpg$", "-pred.jpg", fname)
        cv.imwrite(outfile_path, frame)

def main():
    try:

        for fname in sys.argv[1:]:
            print("Processing {}".format(fname))
            with open(fname, 'rb') as fobj:
                files = {'img': fobj }
                r = session.post(url='http://127.0.0.1:8080/process', files=files, data={'thr_score': '0.5', 'size': '608'})

                handle_response(fname, r)

    except KeyboardInterrupt:
        print("quit")

if __name__ == '__main__':
    main()


