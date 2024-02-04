""" Wrapper for ML-Model
"""

# from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn import metrics as SkM
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split

from pyapollon.basic_structures import (
    ClassificationModel,
    FeaturesLabelsType,
    FeaturesType,
    LabelsType,
    Model,
)
from pyapollon.structure_classes import MLData, TrainTestData

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
    """Object for Evaluating the performances of a model

    Args:
        data (FeaturesLabelsType): Data in Form (Features in numpy, Labels in numpy)
        params (Dict[str,Union[str,float,int]]): Parameters.
            Has the keys with prefix:
                model__ for model parameter
                fit__ for fit parameter
                predict for predict parameter
        model (_type_): Model either parameter has been passed or not
        random_state (int): Number of seed for the randomness
        metrics (Optional[
            Dict[str, Dict[str, Union[Callable[[FeaturesType], float], str]]]
        ]):
            List of tuple containing metric name and dict with:
              * 'func': metric function,
              * 'type': 'label' if metric is computed for predicted label and 'proba'
                for predicted probability.
            Defaults to None. In case None default metrics will be used
        test_size (Optional[float], optional): Proportion of test data. Defaults to 0.2.
        n_splits_cv (Optional[int], optional): Number of Cross Validation of training data.
            Defaults to 5.
        validation_size (Optional[float], optional):
            Proportion of validation data in training data. Defaults to 0.2.
        model_params_assigned (Optional[bool], optional):
            Flag whether model has been assigne with parameters.
            Defaults to False.
    """

    def __init__(
        self,
        data: FeaturesLabelsType,
        params: Dict[str, Union[str, float, int]],
        model: Union[Model, ClassificationModel],
        random_state: int,
        metrics: Optional[
            Dict[str, Dict[str, Union[Callable[[FeaturesType], float], str]]]
        ] = None,
        test_size: Optional[float] = 0.2,
        n_splits_cv: Optional[int] = 5,
        validation_size: Optional[float] = 0.2,
        model_params_assigned: Optional[bool] = False,
    ) -> None:
        """Initialization"""
        self.test_size = test_size
        self.n_splits_cv = n_splits_cv
        self.validation_size = validation_size

        self._params_prefix = ["model", "fit", "predict", "predict_proba"]

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
        self.metrics_performance = {}
        self.val_result = {}
        self.cv_result = {}

        # Whether probability label is predicted
        self.proba_involved = False

        self._set_additional_calls()

    def _set_additional_calls(self) -> None:
        """Method for subclasses executing additional method
        in init. If no method should be executed. Impelent pass
        """
        raise NotImplementedError

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

    def _fit_predict(
        self,
        fit_data: FeaturesLabelsType,
        predict_data: List[FeaturesType],
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

        return list(
            map(
                lambda x: {
                    "label": self.model.predict(x, **self._params["predict"]),
                    "proba": (
                        None
                        if not self.proba_involved
                        else self.model.predict_proba(
                            x, **self._params["predict_proba"]
                        )[:, 1]
                    ),
                },
                predict_data,
            )
        )

    def _compute_metrics(
        self,
        labels_true: LabelsType,
        labels_pred: LabelsType,
        probas_pred: Optional[LabelsType] = None,
        prefix: Optional[str] = None,
    ) -> None:
        _prefix = ""
        if prefix is not None:
            _prefix = f"{prefix}__"
        _result = {}
        for _metric_name, _metric_dict in self.metrics.items():
            _key = _prefix + _metric_name
            if _metric_dict["type"] == "label":
                _result[_key] = _metric_dict["func"](labels_true, labels_pred)
            if _metric_dict["type"] == "proba":
                _result[_key] = _metric_dict["func"](labels_true, probas_pred)
        return _result

    def _evaluate_train_test(  # pylint: disable=dangerous-default-value
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
        _list_predicted = self._fit_predict(
            fit_data=train_test_data_set.train_data.tuple,
            predict_data=train_test_data_set.feature,
        )
        for _name, _label_true, _predicted in zip(
            prefix, train_test_data_set.label, _list_predicted
        ):
            _result_metrics = self._compute_metrics(
                **self._create_arg_compute_metrics(_label_true, _predicted, _name)
            )
            self.metrics_performance.update(_result_metrics)

    def _create_arg_compute_metrics(self, labels_true, predicted, prefix=None):
        _arg_compute_metrics = dict(
            labels_true=labels_true, labels_pred=predicted["label"], prefix=prefix
        )
        if self.proba_involved:
            _arg_compute_metrics = {
                **_arg_compute_metrics,
                "probas_pred": predicted["proba"],
            }
        return _arg_compute_metrics

    def evaluate_with_validation(self) -> None:
        """Evaluate model by validation"""

        self._evaluate_train_test(
            train_test_data_set=self.data.validation_data_set,
            prefix=["sub_train", "sub_val"],
        )

    def _calculate_cv_summary(self, cv_results: Dict):
        cv_summary = {}
        for metric_name in self.metrics:
            metric_values = cv_results[metric_name]
            cv_summary[f"cv__{metric_name}__mean"] = np.mean(metric_values)
            cv_summary[f"cv__{metric_name}__std"] = np.std(metric_values)
        return cv_summary

    def evaluate_with_cv(self) -> None:
        """Evaluate model by cross validation"""
        cv_results = {_metric_name: [] for _metric_name in self.metrics}

        for _data_set in self.data.cv_data_set:

            _predicted = self._fit_predict(
                fit_data=_data_set.train_data.tuple,
                predict_data=[_data_set.test_data.feature],
            )[0]

            _result = self._compute_metrics(
                **self._create_arg_compute_metrics(
                    _data_set.test_data.label, _predicted
                )
            )

            for _key, _value in _result.items():
                cv_results[_key].append(_value)
        self.metrics_performance.update(self._calculate_cv_summary(cv_results))

    def train_model(self) -> None:
        """Train the model respective to train data
        and save the performances for train and test data
        """
        self._evaluate_train_test(train_test_data_set=self.data.train_test_data_set)


class RegressionModelEvaluator(ModelEvaluator):
    """Object for Evaluating the performances of a Regression Model"""

    def _set_additional_calls(self):
        if self.metrics is None:
            self.metrics = {
                "rmse": {"func": SkM.root_mean_squared_error, "type": "label"},
                "mae": {"func": SkM.mean_absolute_error, "type": "label"},
                "maxe": {"func": SkM.max_error, "type": "label"},
                "medae": {"func": SkM.median_absolute_error, "type": "label"},
                "r2": {"func": SkM.r2_score, "type": "label"},
            }


class ClassificationModelEvaluator(ModelEvaluator):
    """Object for Evaluating the performances of a Regression Model"""

    def _set_additional_calls(self):
        self.proba_involved = hasattr(self.model, "predict_proba")
        if self.metrics is None:
            self.metrics = {
                "accuracy": {"func": accuracy_score, "type": "label"},
                "precision": {"func": precision_score, "type": "label"},
                "recall": {"func": recall_score, "type": "label"},
                "f1": {"func": f1_score, "type": "label"},
            }

            if self.proba_involved:
                self.metrics = {
                    **self.metrics,
                    "aucroc": {"func": roc_auc_score, "type": "proba"},
                }
