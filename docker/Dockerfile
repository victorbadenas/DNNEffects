FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get install -y libsndfile1-dev
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
