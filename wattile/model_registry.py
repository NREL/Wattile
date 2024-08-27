from functools import wraps

import mlflow
import requests


def check_registry_availability(func_name, func):
    """
    Decorator to check if the model registry is available before running a
    method
    :param func_name: (str) Name of the function
    :param func: (function) Function to wrap
    :return: (function) Wrapped function
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.log:
            return

        try:
            requests.get(self.registry_endpoint)
            # resetting available flag to True could result in 'dead' runs in
            # the registry
            # self.available = True
        except requests.exceptions.ConnectionError:
            self.available = False

        if not self.available:
            print(f"Model registry not available. Skipping {func_name}.")
            return
        return func(self, *args, **kwargs)

    return wrapper


class ModelRegistryMeta(type):
    """
    Metaclass to wrap all methods in the ModelRegistry class with a decorator
    that checks if the model registry is available before running the method
    :param type: (type) Type of the class
    """

    def __new__(cls, name, bases, dct):
        for attr in dct:
            if callable(dct[attr]):
                if not attr.startswith("_"):
                    dct[attr] = check_registry_availability(attr, dct[attr])

        return super().__new__(cls, name, bases, dct)


class ModelRegistry(metaclass=ModelRegistryMeta):
    """
    Model registry class to log runs, metrics, parameters, and artifacts to the
    model registry
    :param metaclass: (type) Metaclass to wrap all methods in the class
    """

    available = False

    def __init__(self, configs):
        """
        Initialize the model registry
        :param configs: (dict) Dictionary of configurations
        """
        self.log = self._get_val_from_config(configs, ["model_registry", "log"])
        self.registry_endpoint = self._get_val_from_config(
            configs, ["model_registry", "endpoint"], "http://localhost:5000"
        )
        self.experiment_name = self._get_val_from_config(
            configs, ["model_registry", "experiment_name"], "Building ___x___"
        )
        self.run_name = self._get_val_from_config(
            configs, ["model_registry", "run_name"], "Run ___x___"
        )
        self.run_description = self._get_val_from_config(
            configs, ["model_registry", "run_description"], ""
        )
        self.run_tags = self._get_tags(configs)

        try:
            mlflow.set_tracking_uri(self.registry_endpoint)
            requests.get(self.registry_endpoint)
            self.available = True
        except requests.exceptions.ConnectionError as e:
            print(f"Error setting tracking uri: {e}")

    def _get_val_from_config(self, configs, keys, default=None):
        """
        Get a value from a nested dictionary using a list of keys, ensures
        that the key exists before trying to access it to avoid KeyError
        :param configs: (dict) Dictionary to search
        :param keys: (list) List of keys
        :param default: (any) Default value to return if key not found
        :return: (any) Value of the key

        """
        retVal = configs
        for k in keys:
            if k in retVal:
                retVal = retVal[k]
            else:
                return default
        return retVal

    def _get_tags(self, configs):
        """
        Get tags from the configuration, adding the architecture type, version,
        and variant for searchability
        :param configs: (dict) Dictionary of configurations
        :return: (dict) Dictionary of tags
        """
        arch_type = self._get_val_from_config(
            configs, ["learning_algorithm", "arch_type"]
        )
        arch_version = self._get_val_from_config(
            configs, ["learning_algorithm", "arch_version"]
        )
        arch_type_variant = self._get_val_from_config(
            configs, ["learning_algorithm", "arch_type_variant"]
        )
        run_tags = self._get_val_from_config(
            configs, ["model_registry", "run_tags"], {}
        )
        return {
            **run_tags,
            "arch_type": arch_type,
            "arch_version": arch_version,
            "arch_type_variant": arch_type_variant,
        }

    def start_run(self):
        """
        Sets the experiment and starts a run in the model registry
        :param run_name: (str) Name of the run
        :return: None
        """
        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(
                run_name=self.run_name,
                description=self.run_description,
                tags=self.run_tags,
            )
        except Exception as e:
            print(f"Error starting run: {e}")

    def end_run(self):
        """
        Ends the run in the model registry
        :return: None
        """
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Error ending run: {e}")

    def log_input(self, input_data, context):
        """
        Log input data to the model registry
        Note: Max input size is 65k, will fail to log beyond that, we'll need to
        handle this in the future
        :param input_data: (dict) Dictionary of input data
        :param context: (dict) Dictionary of context
        :return: None
        """
        try:
            mlflow.log_input(mlflow.data.from_pandas(input_data), context)
        except Exception as e:
            print(f"Error logging input: {e}")

    def log_metric(self, metric_name, metric_value, step=None):
        """
        Log a metric to the model registry
        :param metric_name: (str) Name of the metric
        :param metric_value: (float) Value of the metric
        :return: None
        """
        try:
            mlflow.log_metric(metric_name, metric_value, step)
        except Exception as e:
            print(f"Error logging metric: {e}")

    def log_param(self, param_name, param_value):
        """
        Log a parameter to the model registry
        :param param_name: (str) Name of the parameter
        :param param_value: (str) Value of the parameter
        :return: None
        """
        try:
            mlflow.log_param(param_name, param_value)
        except Exception as e:
            print(f"Error logging parameter: {e}")

    def log_artifact(self, local_path, artifact_path):
        """
        Log an artifact to the model registry
        :param local_path: (str) Path to the file to write
        :param artifact_path: (str) Path to the artifact
        :return: None
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Error logging artifact: {e}")

    def log_artifacts(self, local_dir, artifact_dir):
        """
        Log all artifacts in a directory to the model registry
        :param local_dir: (str) Path to the directory of artifacts to write
        :param artifact_dir: (str) Path to the directory of artifacts
        :return: None
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_dir)
        except Exception as e:
            print(f"Error logging artifacts: {e}")

    def log_model(self, model, model_name):
        """
        Log a model to the model registry
        :param model: (torch.nn.Module) Model to log
        :param model_name: (str) Name of the model
        :return: None
        """
        try:
            mlflow.pytorch.log_model(model, model_name)
        except Exception as e:
            print(f"Error logging model: {e}")
