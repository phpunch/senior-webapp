import falcon, json
from waitress import serve
from falcon.http_status import HTTPStatus

class HandleCORS(object):
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Max-Age', 1728000)  # 20 days
        if req.method == 'OPTIONS':
            raise HTTPStatus(falcon.HTTP_200, body='\n')

class ObjRes(object):
    __json_content = None

    def __validate_json_input(self, req):
        try:
            self.__json_content = json.loads(req.stream.read())
            print('json from client is validated!')
            return True
        except ValueError, e:
            self.__json_content = {}
            print('json from client is not validated!')
            return False

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = json.dumps({
            'msg': 'hi'
        })
    def on_post(self, req, resp):
        resp.status = falcon.HTTP_200
        validated = self.__validate_json_input(req)

        output = {
            'status': 200,
            'msg': None
        }

        if (validated):
            if ('x' in self.__json_content and 'y' in self.__json_content):
                data = self.__json_content
                equal = int(data['x']) + int(data['y'])
                output = {
                    'msg': 'x: {} + y: {} is equal to {}'.format(data['x'], data['y'], equal)
                }
            else:
                output['status'] = 404
                output['msg'] = 'json input has incorrect params'
        else:
            output['status'] = 404
            output['msg'] = 'json input is not validated'

        resp.body = json.dumps(output)

        resp.body = json.dumps(output)
    def on_put(self, req, resp):
        resp.status = falcon.HTTP_200
        output = {
            'msg': 'not supported for now'
        }
        resp.body = json.dumps(output)
    def on_delete(self, req, resp):
        resp.status = falcon.HTTP_200
        output = {
            'msg': 'not supported for now'
        }
        resp.body = json.dumps(output)

class AudioStorage(object):
    # def on_get(self, req, resp):
    #     print('get')
    #     output = {
    #         'method': 'get'
    #         'user-id': None
    #     }

    def on_post(self, req, resp):
        print(params)
        print('post')
        output = {
            'method': 'post',
        }
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(output)

app = falcon.API(middleware=[HandleCORS()])
app.add_route('/', ObjRes())
app.add_route('/upload', AudioStorage())

serve(app, host="127.0.0.1", port=8000)