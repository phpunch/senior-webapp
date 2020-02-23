import youtube_dl
from pydub import AudioSegment
import pandas as pd
import numpy as np
import subprocess
import os

# video_url = "https://www.youtube.com/watch?v=Y70hiPNxe4Q"
# options = {
#   'format': 'bestaudio/best',
#   'extractaudio' : True,  # only keep the audio
#   'audioformat' : "wav",  # convert to wav
#   'outtmpl': 'demo.wav',    # name the file the ID of the video
#   'noplaylist' : True,    # only download single song, not playlist
# }
# with youtube_dl.YoutubeDL(options) as ydl:
#     ydl.download([video_url])

if (not os.path.exists("audios")):
  os.mkdir("audios")

cmd_string = 'ffmpeg -i demo.wav -f segment -segment_time 3 -ac 1 audios/demo_%06d.wav'
try:
  subprocess.check_call(cmd_string, shell=True)
except subprocess.CalledProcessError as exc:
  print("Fail {}".format(exc.output))

