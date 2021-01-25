FROM tensorflow/tensorflow:2.2.2-gpu-py3

RUN apt-get install -y libsndfile1-dev
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
