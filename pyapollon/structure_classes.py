"""Classes and Structures related to ML Data
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from pyapollon.utils import train_test_split_sorted


class FeatureLabelData:
    """Class for data of form (Feature, Label)"""

    def __init__(self, feature: np.ndarray, label: np.ndarray) -> None:
        """Initialization
        Args:
            feature (np.ndarray): Feature data (Matrix)
            label (np.ndarray): Label data
        """

        if feature.shape[0] != label.shape[0]:
            raise ValueError(
                f"""The number of feature data ({feature.shape[0]})"""
                f"""and label data ({label.shape[0]} differ!"""
            )

        self.feature = feature
        self.label = label
        self.tuple = (feature, label)

    def take_sub_data(self, list_index: List[int]) -> FeatureLabelData:
        """Given an index list. Extract subdata based on this index
        Args:
            list_index (List[int]): List of indices to be extracted

        Returns:
            FeatureLabelData: Extracted data
        """
        return FeatureLabelData(self.feature[list_index], self.label[list_index])

    def split_data_train_test(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = 30,
        index_train_test: Optional[Tuple] = None,
    ) -> TrainTestData:
        """Split data into train and test data
        Args:
            test_size (Optional[float], optional): Proportion test set in the data.
                Defaults to None.
            random_state (Optional[int], optional): Random seed. Defaults to 30.
            index_train_test (Optional[Tuple], optional): Index of the training and test set
                in the data (idx_train,idx_set). Defaults to None.

        Raises:
            ValueError: Error if both test_size and index_train_test is given

        Returns:
            TrainTestData: Splitted data
        """
        if test_size is not None and index_train_test is not None:
            raise ValueError(
                "Either you provide the test size or give the indexes. But not both!"
            )
        if test_size is not None:
            _data_splitted = train_test_split_sorted(
                self.tuple, test_size, random_state
            )
            _data_splitted = [
                FeatureLabelData(*_data_splitted[0]),
                FeatureLabelData(*_data_splitted[1]),
            ]
        if index_train_test is not None:
            _data_splitted = [
                self.take_sub_data(index_train_test[0]),
                self.take_sub_data(index_train_test[1]),
            ]

        return TrainTestData(_data_splitted)


class ListFeatureLabelData:
    """List of FeatureLabelData"""

    def __init__(self, list_data: List[FeatureLabelData]) -> None:
        """Initialization
        Args:
            list_data (List[FeatureLabelData]): List of Feature Label Data
        """
        self.list_data = list_data
        self.feature = [_data.feature for _data in self.list_data]
        self.label = [_data.label for _data in self.list_data]
        self.tuple = [_data.tuple for _data in self.list_data]


class TrainTestData(ListFeatureLabelData):
    """Data for Train and Testing"""

    def __init__(self, list_data: List[FeatureLabelData]):
        """Initialization
        Args:
            list_data (List[FeatureLabelData]): List of data containing training and test data
        """
        super().__init__(list_data)
        self.train_data = list_data[0]
        self.test_data = list_data[1]


class MLData(FeatureLabelData):
    """Data for ML-Model experiment"""

    def __init__(
        self,
        feature: np.ndarray,
        label: np.ndarray,
        test_size: float,
        validation_size: float,
        n_splits_cv: int,
        random_state: int,
    ) -> None:
        """Initialization
        Args:
            feature (np.ndarray): Feature data
            label (np.ndarray): Label data
            test_size (float): Portion of testing data
            validation_size (float): Portion of validation data in the training data
            n_splits_cv (int): Number of CV Splits of training data
            random_state (int): Random state
        """
        super().__init__(feature, label)
        self.test_size = test_size
        self.random_state = random_state

        # Train-Test Data
        self.train_test_data_set = self.split_data_train_test(
            self.test_size, self.random_state
        )

        self.validation_data_set = (
            self.train_test_data_set.train_data.split_data_train_test(
                validation_size, random_state
            )
        )

        self.kf = KFold(
            n_splits=n_splits_cv, shuffle=True, random_state=self.random_state
        )

        self.cv_data_set = list(
            map(
                lambda list_idx: self.train_test_data_set.train_data.split_data_train_test(
                    index_train_test=list_idx
                ),
                self.kf.split(*self.train_test_data_set.train_data.tuple),
            )
        )
