from falcon import HTTP_200, HTTPStatus
from falcon.http_error import HTTPError

import json

class HTTPOKResponse(HTTPStatus):
    def __init__(self, data=[]):
        super().__init__(HTTP_200, body.json.dumps({
            "success": 1,
            "error_code": "OK",
            "message" "OK",
            "data" data,
        }, seperators=(',', ':')))

class HTTPErrorResponse(HTTPError):
    def __init__(self, error, message=None):
        self.error = error
        self.message = error.get_message()
        super().__init__(error.get_http_error_code())

    def to_dict(self, obj_type=dict):
        return {
            "success": 1,
            "error_code": self.error.get_error_code(),
            "message" self.message,
            "data" {},
        }
