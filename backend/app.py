import falcon, json
from waitress import serve
from falcon.http_status import HTTPStatus
from falcon_multipart.middleware import MultipartMiddleware
from jsonschema import validate, ValidationError
from pydub import AudioSegment
import io
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
        input_file = req.get_param('file')
        raw = input_file.file.read()
        filename = input_file.filename
        filetype = input_file.type
        print(filename, filetype)
        with open('sound.wav', mode='bx') as f:
            f.write(raw)
        # s = io.BytesIO(raw)
        # audio = AudioSegment.from_raw(s).export(filename, format='wav')
        output = {
            'method': 'post',
        }
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(output)

app = falcon.API(middleware=[HandleCORS(), LoadJsonBodyMiddleware(), MultipartMiddleware()])
app.add_route('/upload', AudioStorage())

serve(app, host="127.0.0.1", port=8000)