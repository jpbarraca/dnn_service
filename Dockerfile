FROM debian
MAINTAINER João Paulo Barraca

RUN mkdir -p /opt

RUN wget https://pjreddie.com/media/files/yolov3.weights -O /opt/yolov3.weights
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O /opt/yolov3.cfg
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O /opt/coco.names

COPY dnn_service.py /opt
COPY opencv.tar.gz /opt

RUN tar -zxvf /opt/opencv.tar.gz -C /
RUN ls /usr/local/bin

RUN apt-get update
RUN apt-get install -y python3-cherrypy3 libjpeg62-turbo libpng16-16 libtiff5 libvtk6.3 libglu1-mesa python3-numpy vim

WORKDIR /opt

EXPOSE 80

CMD ["/usr/bin/python3", "/opt/dnn_service.py"] 

