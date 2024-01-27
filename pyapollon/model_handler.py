""" Wrapper for ML-Model
"""

# from __future__ import annotations
from typing import Optional, Dict, Callable, Tuple, Union, List
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics as SkM

from pyapollon.basic_structures import (
    LabelsType,
    FeaturesType,
    FeaturesLabelsType,
)
from pyapollon.structure_classes import (
    TrainTestData,
    MLData,
)

# TODO: Type Model


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


class ModelEvaluator:
    """Object for Evaluating the performances of a model"""

    def __init__(
        self,
        data: FeaturesLabelsType,
        params: Dict[str, Union[str, float, int]],
        model,
        random_state: int,
        metrics: Optional[Dict[str, Callable[[FeaturesType], float]]] = None,
        test_size: Optional[float] = 0.2,
        n_splits_cv: Optional[int] = 5,
        validation_size: Optional[float] = 0.2,
        model_params_assigned: Optional[bool] = False,
    ) -> None:
        """Initialization
        Args:
            data (FeaturesLabelsType): Data in Form (Features in numpy, Labels in numpy)
            params (Dict[str,Union[str,float,int]]): Parameters.
                Has the keys with prefix:
                    model__ for model parameter
                    fit__ for fit parameter
                    predict for predict parameter
            model (_type_): Model either parameter has been passed or not
            random_state (int): Number of seed for the randomness
            metrics (Optional[Dict[str, Callable[[FeaturesType], float]]], optional):
                List of tuple containing metric name and metric function. Defaults to None.
                In case None default metrics will be used
            test_size (Optional[float], optional): Proportion of test data. Defaults to 0.2.
            n_splits_cv (Optional[int], optional): Number of Cross Validation of training data.
                Defaults to 5.
            validation_size (Optional[float], optional):
                Proportion of validation data in training data. Defaults to 0.2.
            model_params_assigned (Optional[bool], optional):
                Flag whether model has been assigne with parameters.
                Defaults to False.
        """
        self.test_size = test_size
        self.n_splits_cv = n_splits_cv
        self.validation_size = validation_size

        self._params_prefix = ["model", "fit", "predict"]

        self.params = params

        if not (isinstance(params, dict) and len(params) == 0):
            self._check_params_ok()

        self._parse_params()

        if model_params_assigned:
            self.model = model
        else:
            self.model = model(**self._params["model"])

        self.random_state = random_state

        self.data = MLData(
            *data,
            test_size=test_size,
            validation_size=validation_size,
            n_splits_cv=n_splits_cv,
            random_state=random_state,
        )

        self.kf = KFold(
            n_splits=self.n_splits_cv, shuffle=True, random_state=self.random_state
        )

        self.metrics = metrics

        if metrics is None:
            self.metrics = {
                "rmse": SkM.root_mean_squared_error,
                "mae": SkM.mean_absolute_error,
                "maxe": SkM.max_error,
                "medae": SkM.median_absolute_error,
                "r2": SkM.r2_score,
            }

        self.metrics_performance = {}

        self.val_result = {}
        self.cv_result = {}

    def _check_params_ok(self) -> bool:
        """Check whether the inputted parameter is ok
        Raises:
            KeyError: No parameter has the desired prefix
        Returns:
            bool: _description_
        """
        _set_prefix = {_key.split("__")[0] for _key in self.params.keys()}
        if len(_set_prefix - set(self._params_prefix)) == len(_set_prefix):
            raise KeyError(
                f"In the params, none of this prefixes occurs {self._params_prefix}"
            )

    def _parse_params(self) -> None:
        """Parse inputted parameters in model, fit, and predict params"""
        self._params = {}
        for _sort in self._params_prefix:
            self._params[_sort] = {
                _param.split("__")[1]: _value
                for _param, _value in self.params.items()
                if _sort == _param.split("__")[0]
            }

    def train_model(self) -> None:
        """Train the model respective to train data 
        and save the performances for train and test data
        """
        self._evaluate_train_test(train_test_data_set=self.data.train_test_data_set)

    def _fit_predict(
        self, fit_data: FeaturesLabelsType, predict_data: List[FeaturesType]
    ) -> List[np.ndarray]:
        """Helping function for training the model and predict based on
        inputted data
        Args:
            fit_data (FeaturesLabelsType): Data for fitting the model
            predict_data (FeaturesLabelsType): Features for prediction
        Returns:
            _type_: _description_
        """
        self.model.fit(*fit_data, **self._params["fit"])

        _result = map(
            lambda x: self.model.predict(x, **self._params["predict"]),
            predict_data,
        )
        return list(_result)

    def _compute_metrics(
        self,
        labels_true: LabelsType,
        labels_pred: LabelsType,
        prefix: Optional[str] = None,
    ) -> None:
        _prefix = ""
        if prefix is not None:
            _prefix = f"{prefix}__"
        return {
            _prefix + _metric_name: _metric(labels_true, labels_pred)
            for _metric_name, _metric in self.metrics.items()
        }

    def _evaluate_train_test(
        self,
        train_test_data_set: TrainTestData,
        prefix: Optional[List[str]] = ["train", "test"],
    ) -> None:
        """Train model on training data then evaluate the performance on training
        and test data
        Args:
            train_test_data_set (TrainTestData): Training and Test Dataset
            prefix (Optional[List[str]], optional): Prefixes for the metrics.
                Defaults to ["train", "test"].
        """
        # Get the prediction for train and validation data
        _list_label_predicted = self._fit_predict(
            fit_data=train_test_data_set.train_data.tuple,
            predict_data=train_test_data_set.feature,
        )
        for _name, _label_true, _label_predicted in zip(
            prefix, train_test_data_set.label, _list_label_predicted
        ):
            _result_metrics = self._compute_metrics(
                labels_true=_label_true,
                labels_pred=_label_predicted,
                prefix=_name,
            )
            self.metrics_performance.update(_result_metrics)

    def evaluate_with_validation(self) -> None:
        """Evaluate model by validation"""

        self._evaluate_train_test(
            train_test_data_set=self.data.validation_data_set,
            prefix=["sub_train", "sub_val"],
        )

    def evaluate_with_cv(self) -> None:
        """Evaluate model by cross validation"""
        cv_results = {_metric_name: [] for _metric_name in self.metrics}

        for _data_set in self.data.cv_data_set:
            _label_predicted = self._fit_predict(
                fit_data=_data_set.train_data.tuple,
                predict_data=[_data_set.test_data.feature],
            )[0]
            _result = self._compute_metrics(
                labels_true=_data_set.test_data.label, labels_pred=_label_predicted
            )

            for _key, _value in _result.items():
                cv_results[_key].append(_value)
        print(cv_results)
        self.metrics_performance.update(self._calculate_cv_summary(cv_results))

    def _calculate_cv_summary(self, cv_results: Dict):
        cv_summary = {}
        for metric_name in self.metrics:
            metric_values = cv_results[metric_name]
            cv_summary[f"cv__{metric_name}__mean"] = np.mean(metric_values)
            cv_summary[f"cv__{metric_name}__std"] = np.std(metric_values)
        return cv_summary
