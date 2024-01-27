"""Module for Basic Types"""

from typing import Tuple
import numpy as np

FeaturesType = np.array
LabelsType = np.array
FeaturesLabelsType = Tuple[FeaturesType, LabelsType]
