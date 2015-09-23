import json
import sys, os
import logging
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, Application, url
from tornado.options import parse_command_line
from tornado import gen

from aimetrics import metrics
from aimetrics.conf import get_conf


class EvalReportHandler(RequestHandler):
    """Generate performance reports of a classifier on a project."""

    def initialize(self, ai_url, username, password):
        self.ai_url = ai_url
        self.username = username
        self.password = password

    @gen.coroutine
    def post(self, client_id, project_id, model_type):
        model_params = self.get_argument('params', default=None)
        model_params = json.loads(model_params) if model_params else None
        auth_user = self.get_argument('user', self.username)
        auth_password = self.get_argument('password', self.username)
        report = yield metrics.remote_classifier_report(self.ai_url,
                model_type, client_id, project_id, model_params=model_params,
                auth_username=auth_user, auth_password=auth_password)
        self.finish(report)


def make_app(conf, deploy_mode):
    """Create the application for this server to run"""

    ai_url = conf['aimetrics'][deploy_mode]['hosts']['ai']['base_url']
    username = conf['aimetrics'][deploy_mode]['username']
    password = conf['aimetrics'][deploy_mode]['password']
    logging.info("Using AI Service: %s" % ai_url)
    debug_mode = deploy_mode == "dev"
    return Application([
        url(r'/report/([\w-]+)/([\w-]+)/([\w-]+)', EvalReportHandler,
            dict(ai_url=ai_url, username=username, password=password))
    ], debug=debug_mode)

def main():
    """Start the server IO Loop"""
    conf = get_conf()
    deploy_mode = os.getenv('AIMETRICS_DEPLOY_MODE', "dev").lower()
    parse_command_line()
    app = make_app(conf, deploy_mode)
    port = int(conf['aimetrics'][deploy_mode]['port'])
    logging.info("Listening on port %d" % port)
    app.listen(port)
    IOLoop.current().start()

if __name__ == '__main__':
    main()
