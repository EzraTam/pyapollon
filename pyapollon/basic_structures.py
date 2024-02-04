"""Module for Basic Types"""

from typing import Tuple

import numpy as np

FeaturesType = np.array
LabelsType = np.array
FeaturesLabelsType = Tuple[FeaturesType, LabelsType]


class Model:
    """Abstract Class for Model
    """
    def fit(self):
        """Method for fitting the model to data
        """
    def predict(self):
        """Method for predicting label from data
        """


class ClassificationModel(Model):
    """Abstract class for Classification Model
    """
    def predict_proba(self):
        """Method for predicting the probability of the model
        """
