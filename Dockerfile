FROM pytorch/pytorch
WORKDIR /workspace
ADD . /workspace
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD [ "python" , "/workspace/app.py" ]
RUN chown -R 8080:8080 /workspace
ENV HOME=/workspace
