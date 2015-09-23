#!/bin/env python
from tornado import gen
from tornado.ioloop import IOLoop

from .. import metrics

@gen.coroutine
def main():
    data = yield metrics.fetch_data("http://localhost:3002/", "bst", "drone",
            auth_username="bst", auth_password="bst")
    print(data)
    print("X.shape:\t" + str(data['X'].shape))
    print("y.shape:\t" + str(data['y'].shape))

if __name__ == "__main__":
    IOLoop.instance().run_sync(main)
