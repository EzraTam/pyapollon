"""Module for Basic Types"""

from typing import Tuple

import numpy as np

FeaturesType = np.array
LabelsType = np.array
FeaturesLabelsType = Tuple[FeaturesType, LabelsType]


class Model:
    def fit(self):
        pass

    def predict(self):
        pass


class ClassificationModel(Model):
    def predict_proba(self):
        pass
