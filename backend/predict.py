import youtube_dl
from pydub import AudioSegment
import numpy as np
import subprocess
import os
import glob
import shutil
import pickle
from pprint import pprint
from find_best_plda import find_best_plda, post_process, save_prediction

class YoutubeExistError(Exception): pass
class AudiosExistError(Exception): pass
class WriteWavError(Exception): pass

def clear_dir(folder_name):
  if (os.path.exists("{}/demo.wav".format(folder_name))):
    os.remove("{}/demo.wav".format(folder_name))
  if (os.path.exists("{}/audios".format(folder_name))):
    shutil.rmtree("{}/audios".format(folder_name))
  if (os.path.exists("{}/data".format(folder_name))):
    shutil.rmtree("{}/data".format(folder_name))
  if (os.path.exists("{}/exp/pvector_net/pvector_prod".format(folder_name))):
    shutil.rmtree("{}/exp/pvector_net/pvector_prod".format(folder_name))

# delete old audios and demo.wav first !!!
def check_exists(folder_name):
  if (os.path.exists("{}/demo.wav".format(folder_name))):
    raise YoutubeExistError
  if (os.path.exists("{}/audios".format(folder_name))):
    raise AudiosExistError

def download_youtube(folder_name, video_url):
  # video_url = "https://www.youtube.com/watch?v=kHk5muJUwuw"
  options = {
    'format': 'bestaudio/best',
    'extractaudio' : True,  # only keep the audio
    'audioformat' : "wav",  # convert to wav
    'outtmpl': '{}/demo.wav'.format(folder_name),    # name the file the ID of the video
    'noplaylist' : True,    # only download single song, not playlist
  }
  with youtube_dl.YoutubeDL(options) as ydl:
      ydl.download([video_url])


def audio_segmentation(folder_name):
  if (not os.path.exists("{}/audios".format(folder_name))):
    os.mkdir("{}/audios".format(folder_name))
  # Audio Segmentation 
  duration = 3 # (3 second per file)
  cmd_string = 'ffmpeg -i {0}/demo.wav -f segment -segment_time {1} -ac 1 {0}/audios/demo-%06d.wav'.format(folder_name, duration)
  try:
    subprocess.check_call(cmd_string, shell=True)
  except subprocess.CalledProcessError as exc:
    print("Fail {}".format(exc.output))

def write_wav_file(folder_name):
  print("Write wav file")
  print("Current Dir", os.getcwd())
  if (not os.path.exists("{}/data".format(folder_name))):
    os.mkdir("{}/data".format(folder_name))
  if (not os.path.exists("{}/data".format(folder_name))):
    raise WriteWavError
  path_list = glob.glob('{}/audios/*'.format(folder_name))
  path_list.sort()
  with open("{}/data/wav.scp".format(folder_name), mode='w') as f:
      for path in path_list:
          name = os.path.basename(path)[:-4]
          abs_path = os.path.abspath(path)
          # print("{0} {1}\n".format(name, abs_path))
          f.write("{0} {1}\n".format(name, abs_path))
def test(folder_name):
  try:
    # change working directory to kaldi because of path.sh problem
    # os.chdir(folder_name)
    # run.sh

    command = "./{0}/run_prod.sh {0}".format(folder_name)
    subprocess.check_call(command, shell=True)

    
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
    
    # os.chdir("..")
    
  except subprocess.CalledProcessError as exc:
    with open('{}/exp/pvector_net/pvector_prod/log/extract.log'.format(folder_name)) as f:
      print(f.read())
    print("Status : FAIL", exc.returncode, exc.output)
    print("CallProcessError")
    raise
    # break # handle errors in the called executable
  except OSError:
    raise
    print("OSERROR")
  except FileNotFoundError:
    print("FileNotFoundError")
    raise

def compute_result(folder_name):
  prediction = find_best_plda(folder_name)
  post_prediction = post_process(prediction)
  save_prediction(post_prediction, folder_name)

def get_result(folder_name):
  with open("{}/prediction.pkl".format(folder_name), "rb") as f:
    prediction = pickle.load(f)
    pprint(prediction)
    return prediction
    
if __name__ == "__main__":
  # Create temp folder
  random_string = "55DDD"
  folder_name = "kaldi_{}".format(random_string)
  shutil.copytree("kaldi", folder_name)

  clear_dir(folder_name)
  check_exists(folder_name)
  download_youtube(folder_name, "https://www.youtube.com/watch?v=kHk5muJUwuw")
  audio_segmentation(folder_name)
  write_wav_file(folder_name)
  test(folder_name)
  compute_result(folder_name)
  get_result(folder_name)
  