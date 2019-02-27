FROM czentye/opencv-video-minimal 
MAINTAINER Diogo Gomes <diogogomes@gmail.com> 

RUN mkdir -p /opt

RUN wget https://pjreddie.com/media/files/yolov3.weights -O /opt/yolov3.weights
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O /opt/yolov3.cfg
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O /opt/coco.names

COPY dnn_service.py /opt

RUN pip install cherrypy

WORKDIR /opt

EXPOSE 8080

CMD ["/usr/bin/python3", "/opt/dnn_service.py"] 

