import youtube_dl
from pydub import AudioSegment
import numpy as np
import subprocess
import os
import glob
import shutil

from kaldi.find_best_plda import find_best_plda

class YoutubeExistError(Exception): pass
class AudiosExistError(Exception): pass

# youtube -> youtube 2 -> test
# delete old audios and demo.wav first !!!
def check_exists():
  if (os.path.exists("demo.wav")):
    raise YoutubeExistError
  if (os.path.exists("audios")):
    raise AudiosExistError

def download_youtube(video_url):
  # video_url = "https://www.youtube.com/watch?v=kHk5muJUwuw"
  options = {
    'format': 'bestaudio/best',
    'extractaudio' : True,  # only keep the audio
    'audioformat' : "wav",  # convert to wav
    'outtmpl': 'demo.wav',    # name the file the ID of the video
    'noplaylist' : True,    # only download single song, not playlist
  }
  with youtube_dl.YoutubeDL(options) as ydl:
      ydl.download([video_url])


def audio_segmentation():
  if (not os.path.exists("audios")):
    os.mkdir("audios")
  # Audio Segmentation 
  duration = 3 # (3 second per file)
  cmd_string = 'ffmpeg -i demo.wav -f segment -segment_time {} -ac 1 audios/demo-%06d.wav'.format(duration)
  try:
    subprocess.check_call(cmd_string, shell=True)
  except subprocess.CalledProcessError as exc:
    print("Fail {}".format(exc.output))

def write_wav_file():
  if (not os.path.exists("kaldi/data")):
    os.mkdir("kaldi/data")
  path_list = glob.glob('audios/*')
  path_list.sort()
  with open("kaldi/data/wav.scp", mode='w') as f:
      for path in path_list:
          name = os.path.basename(path)[:-4]
          abs_path = os.path.abspath(path)
          # print("{0} {1}\n".format(name, abs_path))
          f.write("{0} {1}\n".format(name, abs_path))
def test():
  try:
    # # copy wav.scp to directory
    # command = "cp wav.scp kaldi/data"
    # subprocess.check_call(command, shell=True)

    # change working directory to kaldi because of path.sh problem
    os.chdir("kaldi")
    # run.sh
    command = "./run_prod.sh"
    subprocess.check_call(command, shell=True)

    find_best_plda()
    # command = "sort exp/result_prod.txt > exp/result_sorted.txt"
    # subprocess.check_call(command, shell=True)

    # # read the result
    # with open("exp/result_sorted.txt") as f:
    #   prediction = []
    #   prediction_list = [row.strip() for row in f.readlines()]
    #   for row in prediction_list:
    #       name, _, _, _, _, label, score = row.split(" ")
    #       print(name, label, score)
    #       prediction.append({
    #           "name": name,
    #           "label": label,
    #           "score": score
    #       })
    
    os.chdir("..")
    
  except subprocess.CalledProcessError as exc:
    print("Status : FAIL", exc.returncode, exc.output)
    print("CallProcessError")
    # break # handle errors in the called executable
  except OSError:
    print("OSERROR")
  except FileNotFoundError:
    print("FileNotFoundError")

def clear_dir():
  if (os.path.exists("demo.wav")):
    os.remove("demo.wav")
  if (os.path.exists("audios")):
    shutil.rmtree("audios")
  if (os.path.exists("kaldi/data")):
    shutil.rmtree("kaldi/data")
  if (os.path.exists("kaldi/exp/pvector_net/pvector_prod")):
    shutil.rmtree("kaldi/exp/pvector_net/pvector_prod")

    
if __name__ == "__main__":
  current_path = os.getcwd()
  clear_dir()
  check_exists()
  download_youtube("https://www.youtube.com/watch?v=kHk5muJUwuw")
  audio_segmentation()
  write_wav_file()
  test()
  