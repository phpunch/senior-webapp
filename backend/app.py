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

    def on_post(self, req, resp):

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

app_api = falcon.API(middleware=[HandleCORS(), LoadJsonBodyMiddleware(), MultipartMiddleware()])
app_api.add_route('/upload', AudioStorage())

# serve(app, host="127.0.0.1", port=8000)