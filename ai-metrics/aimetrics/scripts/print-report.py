#!/bin/env python
from tornado import gen
from tornado.ioloop import IOLoop

from .. import metrics

@gen.coroutine
def main():
    data = yield metrics.remote_classifier_report("http://localhost:3002/",
            "bnn", "bst", "drone", auth_username="bst", auth_password="bst",
            model_params={"hiddenLayers":[5,5,5,5]})
    print(data)

if __name__ == "__main__":
    IOLoop.instance().run_sync(main)
