import json

import numpy as np
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, HTTPError, Application, url
from tornado.testing import AsyncTestCase, AsyncHTTPTestCase, gen_test

from aimetrics.estimator import RemoteBSTClassifier
from aimetrics.conf import get_conf

clf_model_id = "c7a90189a72f289bc56890ca8f845cce"

create_success_response = """{
    "id": "%s"
}""" % clf_model_id

create_400_response = """{
    "err": "undefined is not a function"
}"""

destroy_success_response = """{
    "ok": true,
    "id": "%s",
    "rev": "2-739b5dd6e949dfcc0502d96f59e737fd"
}""" % create_success_response

destroy_400_response = """
    "err": "deleted"
}"""

train_success_response = """{
    "error": 0.0049970072228306146,
    "iterations": 4816
}"""

train_400_response_body = """{
    "err": "No training set supplied"
}"""

train_400_response_id = """{
    "err": "missing"
}"""

predict_success_response = """[
    [
        0.9333879658720572
    ],
    [
        0.9333879658720572
    ]
]"""

predict_empty_response_body = """[]"""

predict_400_response_id = """{
    "err": "missing"
}"""

train_X = np.asarray([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

train_y = np.asarray([0, 1, 1, 0])


class MockAICreateHandler(RequestHandler):
    def post(self, model_type):
        if model_type == "bnn":
            self.finish(create_success_response)
        else:
            self.set_status(400)
            self.finish(create_400_response)

class MockAIObjectHandler(RequestHandler):
    def delete(self, model_id):
        if model_id == clf_model_id:
            self.finish(destroy_success_response)
        else:
            self.set_status(400)
            self.finish(destroy_400_response)

class MockAITrainHandler(RequestHandler):
    def post(self, model_id):
        if model_id == clf_model_id:
            if len(self.request.body) > 0:
                self.finish(train_success_response)
            else:
                self.set_status(400)
                self.finish(train_400_response_body)
        else:
            self.set_status(400)
            self.finish(train_400_response_id)

class MockAIPredictHandler(RequestHandler):
    def post(self, model_id):
        if model_id == clf_model_id:
            if len(self.request.body) > 0:
                self.finish(predict_success_response)
            else:
                self.finish(predict_empty_response_body)
        else:
            self.set_status(400)
            self.finish(predict_400_response_id)


class TestMockRemoteBSTClassifier(AsyncHTTPTestCase):
    """Test the RemoteBSTClassifier against a mock server"""

    def get_app(self):
        app = Application([
            (r'/classifier/create/(\w+)', MockAICreateHandler),
            (r'/classifier/(\w+)', MockAIObjectHandler),
            (r'/classifier/(\w+)/train', MockAITrainHandler),
            (r'/classifier/(\w+)/predict', MockAIPredictHandler),
        ])
        self.clf = RemoteBSTClassifier(self.get_url('/'), "bnn")
        self.assertIsNotNone(self.clf)
        return app

    @gen_test
    def test_mock_create_model(self):
        clf_id = yield self.clf._create_model()
        self.assertEqual(clf_model_id, clf_id)

    @gen_test
    def test_mock_async_fit(self):
        yield self.clf.async_fit(train_X, train_y)
        self.assertIsNotNone(self.clf.training_error)

    def test_mock_fit(self):
        self.clf.fit(train_X, train_y)


class TestDevRemoteBSTClassifier(AsyncTestCase):
    """Test the RemoteBSTClassifier against the dev server"""

    def setUp(self):
        super().setUp()
        conf = get_conf()
        self.model_params = {
                "hiddenLayers": [5, 6, 7, 8],
                "learningRate": 0.4,
        }
        url = conf['aimetrics']['dev']['hosts']['ai']['base_url']
        self.clf = RemoteBSTClassifier(url, "bnn",
                model_params=self.model_params)
        self.assertIsNotNone(self.clf)

    @gen_test
    def test_async_fit(self):
        yield self.clf.async_fit(train_X, train_y)
        # test params
        m = yield self.clf.get_model()
        self.assertEqual(2+ len(self.model_params['hiddenLayers']),
                len(m["model"]["layers"]))

    def tearDown(self):
        if self.clf.model_id:
            self.clf.destroy_model()



