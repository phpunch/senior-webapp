FROM project_3

# --> Add everything in the current directory to our image, in the 'app' folder.
RUN mkdir -p /kaldi/egs/sre16/senior-webapp/backend
ADD . /kaldi/egs/sre16/senior-webapp/backend

# --> Install dependencies
WORKDIR /kaldi/egs/sre16/senior-webapp/backend

# CMD [ "/root/project/bin/python", "predict.py" ]
# pip falcon falcon_multipart jsonschema pydub youtube_dl tensorflow==1.14.0
# apt-get install ffmpeg
# /root/project/bin/python predict.py