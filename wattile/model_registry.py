import mlflow

class ModelRegistry:

    available = False

    def __init__(self, configs):
        self.registry_endpoint = self.get_val_from_config(
            configs, ['model_registry', 'endpoint'])
        self.experiment_name = self.get_val_from_config(
            configs, ['model_registry', 'experiment_name'])
        self.run_name = self.get_val_from_config(
            configs, ['model_registry', 'run_name'])
        self.run_description = self.get_val_from_config(
            configs, ['model_registry', 'run_description'])
        self.run_tags = self.get_val_from_config(
            configs, ['model_registry', 'run_tags'])

        try:
            mlflow.set_tracking_uri(self.registry_endpoint)
            self.available = True
        except Exception as e:
            print(f'Error setting tracking uri: {e}')

    def get_val_from_config(self, configs, keys, default=None):
        '''
        Get a value from a nested dictionary using a list of keys, ensures
        that the key exists before trying to access it to avoid KeyError
        :param configs: (dict) Dictionary to search
        :param keys: (list) List of keys
        :param default: (any) Default value to return if key not found
        :return: (any) Value of the key

        '''
        retVal = configs
        for k in keys:
            if k in retVal:
                retVal = retVal[k]
            else:
                return default
        return retVal

    def start_run(self):
        '''
        Sets the experiment and starts a run in the model registry
        :param run_name: (str) Name of the run
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping run logging.')
            return

        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(
                run_name=self.run_name,
                description=self.run_description,
                tags=self.run_tags)
        except Exception as e:
            print(f'Error starting run: {e}')

    def end_run(self):
        '''
        Ends the run in the model registry
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping run logging.')
            return

        try:
            mlflow.end_run()
        except Exception as e:
            print(f'Error ending run: {e}')

    def log_input(self, input_data, context):
        '''
        Log input data to the model registry
        Note: Max input size is 65k, will fail to log beyond that, we'll need to
        handle this in the future
        :param input_data: (dict) Dictionary of input data
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping input logging.')
            return

        try:
            mlflow.log_input(mlflow.data.from_pandas(input_data), context)
        except Exception as e:
            print(f'Error logging input: {e}')

    def log_metric(self, metric_name, metric_value, step=None):
        '''
        Log a metric to the model registry
        :param metric_name: (str) Name of the metric
        :param metric_value: (float) Value of the metric
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping metric logging.')
            return

        try:
            mlflow.log_metric(metric_name, metric_value, step)
        except Exception as e:
            print(f'Error logging metric: {e}')

    def log_param(self, param_name, param_value):
        '''
        Log a parameter to the model registry
        :param param_name: (str) Name of the parameter
        :param param_value: (str) Value of the parameter
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping parameter logging.')
            return

        try:
            mlflow.log_param(param_name, param_value)
        except Exception as e:
            print(f'Error logging parameter: {e}')

    def log_artifact(self, local_path, artifact_path):
        '''
        Log an artifact to the model registry
        :param artifact_path: (str) Path to the artifact
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping artifact logging.')
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f'Error logging artifact: {e}')

    def log_artifacts(self, local_dir, artifact_dir):
        '''
        Log all artifacts in a directory to the model registry
        :param artifact_dir: (str) Path to the directory of artifacts
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping artifact logging.')
            return

        try:
            mlflow.log_artifacts(local_dir, artifact_dir)
        except Exception as e:
            print(f'Error logging artifacts: {e}')

    def log_model(self, model, model_name):
        '''
        Log a model to the model registry
        :param model: (torch.nn.Module) Model to log
        :param model_name: (str) Name of the model
        :return: None
        '''
        if not self.available:
            print('Model registry not available. Skipping model logging.')
            return

        try:
            mlflow.pytorch.log_model(model, model_name)
        except Exception as e:
            print(f'Error logging model: {e}')
