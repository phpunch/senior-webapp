from __future__ import unicode_literals
from gevent import monkey, pywsgi  # import the monkey for some patching as well as the WSGI server
monkey.patch_all()  # make sure to do the monkey-patching before loading the falcon package!

import falcon, json
# from waitress import serve
from falcon.http_status import HTTPStatus
from falcon_multipart.middleware import MultipartMiddleware
from jsonschema import validate, ValidationError
from pydub import AudioSegment
import io
import os
import subprocess
import os
import pickle
import shutil

from pprint import pprint

import predict
from save_label import save_label


class HandleCORS(object):
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Max-Age', 1728000)  # 20 days
        if req.method == 'OPTIONS':
            raise HTTPStatus(falcon.HTTP_200, body='\n')

class LoadJsonBodyMiddleware(object):
  def process_resource(self, req, resp, resource, params):
    try: 
      params['data'] = json.loads(req.bounded_stream.read())
    except:
      pass


class AudioStorage(object):
    __audio_filename = ""

    def on_post_upload(self, req, resp):

        # Get audio file
        input_file = req.get_param('file')
        raw = input_file.file.read()
        filename = input_file.filename
        filetype = input_file.type
        print(filename, filetype)

        if (os.path.exists(filename)):
          os.remove(filename)
        if (os.path.exists("wav.scp")):
          os.remove("wav.scp")
        if (os.path.exists("../../v13_parliament_v3/exp/pvector_net/pvector_prod")):
          for root, dirs, files in os.walk("../../v13_parliament_v3/exp/pvector_net/pvector_prod", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


        # Write audios
        with open(filename, mode='bx') as f:
            f.write(raw)

        # Write wav.scp
        path = os.path.abspath(os.getcwd())
        with open("wav.scp", mode='w') as f:
            f.write("{0} {1}/{0}\n".format(filename, path))


        try:
            command = "cp wav.scp ../../v13_parliament_v3/test_data"
            subprocess.check_call(command, shell=True)

            command = "../../v13_parliament_v3/run_prod.sh"
            subprocess.check_call(command, shell=True)

            with open("../../v13_parliament_v3/exp/result_prod.txt") as f:
              text = f.read()
              print(text)
            
        except subprocess.CalledProcessError as exc:
          print("Status : FAIL", exc.returncode, exc.output)
          print("CallProcessError")
        except OSError:
          print("OSERROR")
        except FileNotFoundError:
          print("FileNotFoundError")

        output = {
            'method': 'post',
            'predicted_output': text
        }


        resp.status = falcon.HTTP_200
        resp.body = json.dumps(output)

    def on_post_youtube(self, req, resp, data=None):
      print(data)
      try:
        # Create temp folder
        random_string = data["ticket"]
        folder_name = "kaldi_{}".format(random_string)
        if (os.path.exists(folder_name)):
          raise
        shutil.copytree("kaldi", folder_name)

        predict.clear_dir(folder_name)
        predict.check_exists(folder_name)
        predict.download_youtube(folder_name, data["youtube_url"])
        predict.audio_segmentation(folder_name)
        predict.write_wav_file(folder_name)
        predict.test(folder_name)
        predict.compute_result(folder_name)
        prediction = predict.get_result(folder_name)
      except:
        raise Exception
      finally:
        if (os.path.exists(folder_name)):
          shutil.rmtree(folder_name)

      output = {
          'method': 'post',
          'prediction': prediction
      }

      resp.status = falcon.HTTP_200
      resp.body = json.dumps(output)

    def on_post_save(self, req, resp, data=None):
      try:
        print(data)
        # Create temp folder
        random_string = data["ticket"]
        folder_name = "kaldi_{}".format(random_string)
        if (os.path.exists(folder_name)):
          raise
        shutil.copytree("kaldi", folder_name)

        predict.clear_dir(folder_name)
        predict.check_exists(folder_name)
        predict.download_youtube(folder_name, data["youtube_url"])
        predict.audio_segmentation(folder_name)
        predict.write_wav_file(folder_name)
        predict.test(folder_name)
        predict.compute_result(folder_name)
        predict.get_result(folder_name)

        # save label
        save_label(folder_name, data["video_id"], data["label_list"])
      except:
        raise Exception
      finally:
        if (os.path.exists(folder_name)):
          shutil.rmtree(folder_name)
      output = {
          'method': 'post',
          'success': 1
      }

      resp.status = falcon.HTTP_200
      resp.body = json.dumps(output)


app_api = falcon.API(middleware=[HandleCORS(), LoadJsonBodyMiddleware(), MultipartMiddleware()])
app_api.add_route('/upload', AudioStorage(), suffix="upload")
app_api.add_route('/youtube', AudioStorage(), suffix="youtube")
app_api.add_route('/save', AudioStorage(), suffix="save")

port = 8000
print("listening")
server = pywsgi.WSGIServer(("0.0.0.0", port), app_api)  # address and port to bind, and the Falcon handler API
server.serve_forever()  # once the server is created, let it serve forever
# serve(app, host="127.0.0.1", port=8000)