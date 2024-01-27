"""Module for help functions"""

from typing import Tuple
from sklearn.model_selection import train_test_split
from pyapollon.basic_structures import FeaturesLabelsType


def train_test_split_sorted(
    data: FeaturesLabelsType, test_size: int, random_state: int
) -> Tuple[FeaturesLabelsType, FeaturesLabelsType]:
    """Wrapper for train_test_split, s.t. we have data_train, data_test for output
    Args:
        data (FeaturesLabelsType): Data to be splitted.
            Data have the form (features,labels)
        test_size (int): Proportion of test data
        random_state (int): Random State

    Returns:
        Tuple[FeaturesLabelsType,FeaturesLabelsType]: (Train Data, Test Data)
    """
    features_train, features_test, labels_train, labels_test = train_test_split(
        *data, test_size=test_size, random_state=random_state
    )
    return (features_train, labels_train), (features_test, labels_test)
