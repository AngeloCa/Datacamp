import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import  f1_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


problem_title = 'Parisian associations grants prediction challenge'

_target_column_name = 'montant vote'
# Label for binary classification
_prediction_label_names = [1, 0]

# We first need a classifier
Predictions_1 = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
# Then a regressor
Predictions_2 = rw.prediction_types.make_regression(label_names=[_target_column_name])
# The combined Predictions is initalized by the list of individual Predictions.
Predictions = rw.prediction_types.make_combined([Predictions_1, Predictions_2])


class clfreg(object):

    def __init__(self, workflow_element_names=[
            'feature_extractor_clf', 'classifier',
            'feature_extractor_reg', 'regressor']):
        self.element_names = workflow_element_names
        self.feature_extractor_classifier_workflow =\
            FeatureExtractorClassifier(self.element_names[:2])
        self.feature_extractor_regressor_workflow =\
            FeatureExtractorRegressor(self.element_names[2:])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # Avoid setting with copy warning
        X_train_df = X_df.iloc[train_is].copy()
        y_train_array = y_array[train_is].copy()

        y_train_clf = y_train_array[:, 0].copy()
        y_train_reg = y_train_array[:, 1].copy()
        
        idx = np.where(y_train_reg > 0)[0]
        
        y_train_reg = y_train_reg[idx]
        

        fe_clf, clf = self.feature_extractor_classifier_workflow.\
            train_submission(module_path, X_train_df, y_train_clf)

        fe_reg, reg = self.feature_extractor_regressor_workflow.\
            train_submission(module_path, X_train_df.loc[idx,:], y_train_reg)

        return fe_clf, clf, fe_reg, reg

    def test_submission(self, trained_model, X_df):
        fe_clf, clf, fe_reg, reg = trained_model
        y_pred_clf = self.feature_extractor_classifier_workflow.\
            test_submission((fe_clf, clf), X_df)
        
        # Avoid setting with copy warning
        X_df = X_df.copy()
        
        labels = np.argmax(y_pred_clf, axis=1)
    
        # get only subventioned label idx
        pred_idx = np.where(labels != 0)[0]
        y_pred_reg = np.full((y_pred_clf.shape[0],), -1)
        
        
        y_temp = self.feature_extractor_regressor_workflow.\
            test_submission((fe_reg, reg), X_df)

        y_pred_reg[pred_idx] = y_temp[pred_idx]

        return np.concatenate([y_pred_clf, y_pred_reg.reshape(-1, 1)], axis=1)

workflow = clfreg()

class F1_score(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1', precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
      
        
        labels = np.argmax(y_pred, axis=1)    
                
        return 1-f1_score(y_true[:,0], labels)

class log_score(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='fan error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        idx = np.where(y_pred > -1)[0]

        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        max_true = np.maximum(3., np.log10(np.maximum(1., y_true[idx])))
        max_pred = np.maximum(3., np.log10(np.maximum(1., y_pred[idx])))
        
        loss = np.mean(np.abs(max_true - max_pred))
        
        return loss


score_clf = F1_score()
score_reg = log_score()

score_types = [
    #Combination with 0.6, 0.4
    rw.score_types.Combined(
        name='combined', score_types=[score_clf, score_reg],
        weights=[0.6, 0.4], precision=2),
    rw.score_types.MakeCombined(score_type=score_clf, index=0),
    rw.score_types.MakeCombined(score_type=score_reg, index=1),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y, groups=X['numDoc'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                       compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)

    y_reg_array = y_array.copy()
    y_clf_array = y_array.copy()
    y_clf_array[y_clf_array > 0 ] = 1 
    y_array = np.concatenate([y_clf_array.reshape(-1,1), y_reg_array.reshape(-1,1)], axis=1)

    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'subventions-accordees-et-refusees_TRAIN.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'subventions-accordees-et-refusees_TEST.csv'
    return _read_data(path, f_name)
