import json
import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPError, HTTPClient
from tornado.ioloop import IOLoop
from urllib.parse import urljoin

from .conf import get_conf


_conf = get_conf()

# one day in seconds
DEFAULT_TIMEOUT = 86400


class RemoteBSTClassifier(BaseEstimator, ClassifierMixin):
    """ RemoteClassifier is a classifier that relies on a remote BST service to
    make predictions.

    This relies on the the REST interface laid out by the bst.ai server for
    interaction, so it is not generalizable to other servers.

    Parameters
    ----------
    base_url : str
        The base URL of the remote API.

    model_type : str
        The model type to use on the remote API.  Refer to the bst.ai project
        for available options.

    model_params : dict (default: {})
        Any model parameters for the remote classifier. Refer to the bst.ai
        project for available options.

    Attributes
    ----------
    model_id : str or None
        The ID of the remote model, or None if the remote model has not been
        instantiated.

    training_error : float or None
        The training error of the trained classifier, or None if the classifier
        has not been trained yet.

    """

    def __init__(self, base_url, model_type, model_params=None):
        self._conf = _conf['aimetrics']['estimators']['RemoteBSTClassifier']
        self.model_type = model_type
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.base_url = base_url
        self.model_id = None
        self.training_error = None

    @gen.coroutine
    def _create_model(self):
        """Create a new model and return the ID

        This does not set the self.model_id attribute.

        """
        http_client = AsyncHTTPClient()
        # assemble request parameters
        create_url_suffix = self._conf['create']['url_suffix'].format(
                model_type = self.model_type)
        create_url = urljoin(self.base_url, create_url_suffix)
        create_method = self._conf['create']['method']
        create_params = json.dumps(self.model_params)
        headers = {'content-type': 'application/json'}
        # send async create request
        response = yield http_client.fetch(create_url, method=create_method,
                body=create_params, headers=headers,
                connect_timeout=DEFAULT_TIMEOUT,
                request_timeout=DEFAULT_TIMEOUT)
        return json.loads(response.body.decode('utf-8'))['id']

    @gen.coroutine
    def destroy_model(self):
        """Destroy a model"""
        http_client = AsyncHTTPClient()
        # assemble request parameters
        destroy_url_suffix = self._conf['destroy']['url_suffix'].format(
                model_id = self.model_id)
        destroy_url = urljoin(self.base_url, destroy_url_suffix)
        destroy_method = self._conf['destroy']['method']
        headers = {'content-type': 'application/json'}
        # send async destroy request
        response = yield http_client.fetch(destroy_url, method=destroy_method,
                headers=headers, connect_timeout=DEFAULT_TIMEOUT,
                request_timeout=DEFAULT_TIMEOUT)
        self.model_id = None
        return json.loads(response.body.decode('utf-8'))

    @gen.coroutine
    def _train_model(self, training_set, training_params=None):
        """Train the classifier's remote model"""
        if training_params is None:
            training_params = {}
        http_client = AsyncHTTPClient()
        # assemble request parameters
        train_url_suffix = self._conf['train']['url_suffix'].format(
                model_id=self.model_id)
        train_url = urljoin(self.base_url, train_url_suffix)
        train_cmd = { "trainingSet": training_set,
                      "params": training_params }
        train_method = self._conf['train']['method']
        headers = {'content-type': 'application/json'}
        response = yield http_client.fetch(train_url, method=train_method,
                body=json.dumps(train_cmd), headers=headers,
                connect_timeout=DEFAULT_TIMEOUT,
                request_timeout=DEFAULT_TIMEOUT)
        return json.loads(response.body.decode('utf-8'))

    @gen.coroutine
    def _predict_model(self, prediction_set):
        """Predict the classes for a set of records using a remote model"""
        http_client = AsyncHTTPClient()
        # assemble request parameters
        predict_url_suffix = self._conf['predict']['url_suffix'].format(
                model_id=self.model_id)
        predict_url = urljoin(self.base_url, predict_url_suffix)
        predict_method = self._conf['predict']['method']
        headers = {'content-type': 'application/json'}
        try:
            response = yield http_client.fetch(predict_url,
                    method=predict_method, body=json.dumps(prediction_set),
                    headers=headers, connect_timeout=DEFAULT_TIMEOUT,
                    request_timeout=DEFAULT_TIMEOUT)
        except HTTPError as e:
            if e.response and e.response.body:
                logging.error(e.response.body.decode('utf-8'))
            raise e

        return json.loads(response.body.decode('utf-8'))

    @gen.coroutine
    def get_model(self):
        """Get the JSON representation of the model from the server"""
        http_client = AsyncHTTPClient()
        # assemble request parameters
        url_suffix = self._conf['get']['url_suffix'].format(
                model_id=self.model_id)
        url = urljoin(self.base_url, url_suffix)
        method = self._conf['get']['method']
        headers = {'content-type': 'application/json'}
        response = yield http_client.fetch(url, method=method, headers=headers,
               request_timeout=DEFAULT_TIMEOUT)
        return json.loads(response.body.decode('utf-8'))


    @gen.coroutine
    def async_fit(self, X, y):
        """ Asynchronously fit the remote classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values

        """
        # import ipdb; ipdb.set_trace() # DEBUG
        if self.model_id is None:
            self.model_id = yield self._create_model()
        # create training set
        if isinstance(X, np.ndarray):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)
        if isinstance(y, np.ndarray):
            y = y.tolist()
        elif not isinstance(y, list):
            y = list(y)
        training_set = [{'input': x_row, 'output': y_row}
                for x_row, y_row in zip(X,y)]
        # perform training
        train_results = yield self._train_model(training_set)
        # record and return error
        self.training_error = train_results['error']
        return self.training_error

    def fit(self, X, y):
        """Fit the remote classifier.
        NOT YET IMPLEMENTED

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values

        """
        # TODO: Bugfix RemoteBSTClassifier synchronous functions
        """
        def fit_wrapper():
            self.async_fit(X, y)
        IOLoop.instance().run_sync(fit_wrapper)
        """
        raise NotImplementedError

    @gen.coroutine
    def async_predict_proba(self, X):
        """Predict class labels for samples in X"""
        # make sure we have trained a model
        if self.model_id is None:
            raise NotFittedError("This BST Model is not fitted yet.")
        # clean prediction set
        if isinstance(X, np.ndarray):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)
        # perform and return prediction
        results = yield self._predict_model(X)
        return np.asarray(results)
