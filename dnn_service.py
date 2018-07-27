#encoding=utf-8
__author__      = "João Paulo Barraca"
__email__ = "jpbarraca@gmail.com"
__copyright__   = "Copyright 2018, João Paulo Barraca"
__license__ = "GPL"
__status__ = "hack"

import cv2 as cv
import numpy as np
import time
import logging
import cherrypy
import tempfile
import os
import json

class DNNService(object):
    
    def __init__(self, net, classes, outputs_names):
        self.net = net
        self.classes = classes
        self.outputs_names = outputs_names

    @cherrypy.expose
    def index(self):
        return '<html><body><form action="/process" method="post" enctype="multipart/form-data">File: <input type="file" name="img"></input><br />Score Threshold: <input type="text" name="thr_score" value="0.5"><br />NMS Threshold: <input type="text" name="thr_nms" value="0.4"></input><input type="submit" value="Process"></input></form></body></html>'

    @cherrypy.expose
    def process(self, img, thr_score=0.5, thr_nms=0.4, size=320):
        global confidence_threshold
        
        fd, path = tempfile.mkstemp()

        with os.fdopen(fd, "wb") as tmp:
            rsize = 0
            while True:
                data = img.file.read(8192)
                rsize += len(data)
                if not data:
                    break

                if rsize > 20000000:
                    tmp.close()
                    raise cherrypy.HTTPError(status=400)

                tmp.write(data)
            tmp.close()
        try:
            if isinstance(thr_score, str):
                thr_score = float(thr_score)

            if isinstance(thr_nms, str):
                thr_nms = float(thr_nms)
            
            if isinstance(size, str):
                size = int(size)
        except:
            raise cherrypy.HTTPError(status=400)

        if thr_score < 0 or thr_score > 1 or thr_nms < 0 or thr_nms > 1 or size not in [320, 416, 608]: 
            raise cherrypy.HTTPError(status=400)

        cap = cv.VideoCapture(path)
        
        os.remove(path)

        has_frame, frame = cap.read()
        if not has_frame:
            raise cherrypy.HTTPError(status=400)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        blob = cv.dnn.blobFromImage(frame, 1/255, (size, size), 0, True, crop=False)

       
        tstart = time.time()
        net.setInput(blob)
        outs = net.forward(self.outputs_names)
        tend= time.time()

        print('Inference time: %.2f s' % (tend - tstart))

        result = self.postprocess(net, classes, frame, outs, thr_score, thr_nms, frame_width, frame_height)

        cherrypy.response.headers['Content-Type'] = 'text/json'
        
        return json.dumps(result, indent=4)        
        
    def postprocess(self, net, classes, frame, outs, thr_score, thr_nms, frame_width, frame_height):
        predictions = []

        layer_names = net.getLayerNames()
        last_layer_id = net.getLayerId(layer_names[-1])
        last_layer = net.getLayer(last_layer_id)

        classIds = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence >= thr_score:
                    print(confidence)
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
            
        indices = cv.dnn.NMSBoxes(boxes, confidences, thr_score, thr_nms)

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
        
            predictions.append( {'class': str(classes[classIds[i]]), 'box': {'x': left, 'y': top, 'x1': left + width, 'y1': top + height}, 'confidence': round(float(confidences[i]),3)})
        
        return predictions


if __name__ == '__main__':
    print("Creating Net")
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    
    ln = net.getLayerNames()
    outputs_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    with open('coco.names', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    print("Starting DNN Service")

    cherrypy.server.socket_host = '0.0.0.0' 
    cherrypy.quickstart(DNNService(net, classes, outputs_names))
    
