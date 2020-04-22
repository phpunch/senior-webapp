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
import kaldi_io

class YoutubeExistError(Exception): pass
class AudiosExistError(Exception): pass
class WriteWavError(Exception): pass
class AudiosDisappearError(Exception): pass

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
  cmd_string = 'ffmpeg -i {0}/demo.wav -f segment -segment_time {1} -ar 16000 -ac 1 {0}/audios/demo-%06d.wav'.format(folder_name, duration)
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


def tune_loudness(folder_name):
  try:
    command = "./{0}/run_vad.sh {0}".format(folder_name)
    subprocess.check_call(command, shell=True)

    vec = {k: v.astype(int) for k,v in kaldi_io.read_vec_flt_ark("{}/mfcc/vad.ark".format(folder_name))}
    # print(vec)
    for filename in vec.keys():
      file_path = os.path.join(folder_name, 'audios', '{}.wav'.format(filename))
      sound = AudioSegment.from_file(file_path)

      target_dBFS = -10
      change_in_dBFS = target_dBFS - sound.dBFS
      modified_sound = sound.apply_gain(change_in_dBFS)
      modified_sound.export(file_path, format="wav")
  except subprocess.CalledProcessError as exc:
    print("Status : FAIL", exc.returncode, exc.output)
    print("CallProcessError")
    raise

def test(folder_name):
  try:
    command = "./{0}/run_prod.sh {0}".format(folder_name)
    subprocess.check_call(command, shell=True)
    
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

def compute_result(folder_name, video_id):
  prediction = find_best_plda(folder_name)
  # prediction = post_process(prediction)
  save_prediction(prediction, folder_name, video_id)

def get_result(folder_name):
  with open("{}/prediction.pkl".format(folder_name), "rb") as f:
    prediction = pickle.load(f)
    pprint(prediction)
    return prediction
    
if __name__ == "__main__":
  # Create temp folder
  random_string = "adapt_4"
  folder_name = "kaldi_{}".format(random_string)
  shutil.copytree("kaldi", folder_name)

  clear_dir(folder_name)
  check_exists(folder_name)
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=kHk5muJUwuw")
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=TmZDlDTK03w")
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=2_UQRXojFWw")
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=WItn14CMDSM") # 1
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=3d0uirQAzvc&feature=youtu.be") # 2 26
  # download_youtube(folder_name, "https://www.youtube.com/watch?v=JEIOIM80mk8&feature=youtu.be") # 3 24
  download_youtube(folder_name, "https://www.youtube.com/watch?v=yMvHWLLvfvc&feature=youtu.be") # 4 25.1
  audio_segmentation(folder_name)
  write_wav_file(folder_name)
  tune_loudness(folder_name)
  raise Exception
  test(folder_name)
  compute_result(folder_name, 'test')
  get_result(folder_name)

  