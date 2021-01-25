FROM tensorflow/tensorflow:2.2.2-gpu-py3

RUN apt-get install -y libsndfile1-dev
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install pandas \
                soundfile \
                jupyter \
                matplotlib
