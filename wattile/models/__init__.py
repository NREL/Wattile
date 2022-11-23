from wattile.error import ConfigsError
from wattile.models.alfa_model import AlfaModel
from wattile.models.AlgoMainRNNBase import AlgoMainRNNBase
from wattile.models.bravo_model import BravoModel
from wattile.models.charlie_model import CharlieModel

MODELS_DICT = {"alfa": AlfaModel, "bravo": BravoModel, "charlie": CharlieModel}


class ModelFactory:
    @staticmethod
    def create_model(configs: dict) -> AlgoMainRNNBase:
        """create model

        :param configs: configs
        :type configs: dict
        :raises ConfigsError: if arch versions is invalid
        :return: new model
        :rtype: AlgoMainRNNBase
        """
        arch_version = configs["learning_algorithm"]["arch_version"]
        if arch_version == "alfa":
            return AlfaModel(configs)

        elif arch_version == "bravo":
            return BravoModel(configs)

        else:
            raise ConfigsError(
                "ModelFactory can only accept arch versions alfa and bravo."
            )
