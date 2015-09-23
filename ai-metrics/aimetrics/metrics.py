import json
import numpy as np
from sklearn.cross_validation import (StratifiedShuffleSplit, StratifiedKFold,
        KFold)
from sklearn.preprocessing import binarize, normalize
from sklearn.metrics import (accuracy_score, roc_curve, roc_auc_score,
                             f1_score, classification_report)
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPError, HTTPClient
from urllib.parse import urljoin

from .conf import get_conf
from .estimator import RemoteBSTClassifier

_conf = get_conf()['aimetrics']['metrics']

@gen.coroutine
def fetch_data(base_url, client_id, project_id, **kwargs):
    """Fetch all labeled records from cloudant for a project

    Arguments
    ---------
        base_url : str
        client_id : str
        project_id : str
        **kwargs : **dict
            Any additional keyword arguments are passed to
            tornado.httpclient.AsyncHTTPClient.fetch.  This is where
            authentication credentials can be specified.

    """
    http_client = AsyncHTTPClient()
    url_suffix = _conf['data']['url_suffix'].format(client_id=client_id,
            project_id=project_id)
    url = urljoin(base_url, url_suffix)
    method = _conf['data']['method']
    response = yield http_client.fetch(url, method=method, **kwargs)
    data = json.loads(response.body.decode('utf-8'))
    features = data[0]['input'].keys()
    classes = data[0]['output'].keys()
    # print("DATA: " + response.body.decode('utf-8'))
    X = np.asarray([[row['input'][k] for k in features] for row in data])
    y = np.asarray([[row['output'].get(k, 0) for k in classes] for row in data])
    return {
            "features": features,
            "classes": classes,
            "X": X,
            'y': y,
    }

@gen.coroutine
def remote_classifier_report(base_url, model_type, client_id, project_id,
        model_params=None, auth_username=None, auth_password=None,
        threshold=0.5, destroy_model=True, save_model=False):
    """Evaluate model performances on a specific BST project dataset.

    Performs 5-fold cross-validation using all classified records from the
    provided client and project ID, and returns a list of evaluation metrics
    from each run.

    Parameters
    ----------
    base_url : str
        the base URL of the remote API.

    model_type : str
        The model type to use on the remote API.  Refer to the bst.ai project
        for available options.

    client_id : str
        The client's BlackSage Tech ID

    project_id : str
        The client's project BlackSage Tech ID

    auth_username : str (default: None)
        The username to use for basic authentication.

    auth_password : str (default: None)
        The password to use for basic authentication.

    model_params : dict (default: {})
        Any model parameters for the remote classifier. Refer to the bst.ai
        project for available options.

    threshold : float (default: 0.5)
        The threshold at which to consider a probability prediction a positive
        classification for use in metrics which take binary input.

    destroy_model : boolean (default: True)
        If True, the trained remote model is destroyed after evaluation is
        complete.

    save_model : boolean (default: False)
        If true, a serialization of the model is attached to the output
        dictionary under the key `model`.

    """
    data = yield fetch_data(base_url, client_id, project_id,
            auth_username=auth_username, auth_password=auth_password)
    X, y = normalize(data['X']), normalize(data['y'])
    # import ipdb; ipdb.set_trace() # DEBUG
    """
    tv_ind, test_ind = StratifiedShuffleSplit(y, 1, 0.2)[0]
    X_tv, X_test = X[tv_ind], X[test_ind]
    y_tv, y_test = y[tv_ind], y[test_ind]
    """
    #skf = StratifiedKFold(y, 5, True)
    skf = KFold(y.shape[0], 5, True)
    cv_results = []
    for train_ind, test_ind in skf:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        result = yield remote_classifier_metrics(base_url, model_type, X_train,
                y_train, X_test, y_test, data['classes'],
                model_params=model_params, destroy_model=destroy_model,
                threshold=threshold, save_model=save_model)
        cv_results.append(result)
    return {"cross_validation": cv_results}

@gen.coroutine
def remote_classifier_metrics(base_url, model_type, X_train, y_train, X_test,
        y_test, data_classes, model_params=None, destroy_model=True,
        threshold=0.5, save_model=False):
    """Train and evaluate a single model with the provided data.

    Parameters
    ----------
    base_url : str
        the base URL of the remote API.

    model_type : str
        The model type to use on the remote API.  Refer to the bst.ai project
        for available options.

    X_train : np.ndarray
        Training feature vectors

    y_train : np.ndarray
        Training target vectors

    X_test : np.ndarray
        Testing feature vectors

    y_test : np.ndarray
        Testing target vectors

    data_classes : [str..]
        Class labels for y targets

    model_params : dict (default: {})
        Any model parameters for the remote classifier. Refer to the bst.ai
        project for available options.

    destroy_model : boolean (default: True)
        If True, the trained remote model is destroyed after evaluation is
        complete.

    threshold : float (default: 0.5)
        The threshold at which to consider a probability prediction a positive
        classification for use in metrics which take binary input.

    save_model : boolean (default: False)
        If true, a serialization of the model is attached to the output
        dictionary under the key `model`.


    Returns: A dictionary of evaluation metrics for the trained model.
    """
    # create a new classifier and object to store results
    clf = RemoteBSTClassifier(base_url, model_type, model_params=model_params)
    result = {}
    try:
        result['train_error'] = yield clf.async_fit(X_train, y_train)
        y_pred_proba = yield clf.async_predict_proba(X_test)
        if save_model:
            result['model'] = yield clf.get_model()
        y_pred = binarize(y_pred_proba, threshold)
        result['acc'] = accuracy_score(y_test, y_pred)
        result['f1_score'] = f1_score(y_test, y_pred)
        result['classification_report'] = classification_report(y_test, y_pred)
        roc= {}
        for i, label in enumerate(data_classes):
            y_test_i = y_test[:,i]
            # skip tests with no actual values
            if np.sum(y_test_i) == 0:
                continue
            fpr, tpr, thresh = roc_curve(y_test[:,i], y_pred_proba[:,i])
            roc[label] = {
                "fpr": list(fpr),
                "tpr": list(tpr),
                "threshold": list(thresh),
            }
        result['roc'] = roc
        try:
            result['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            result['roc_auc'] = None
    finally:
        if(destroy_model):
            yield clf.destroy_model()
    return result
