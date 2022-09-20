
from Model import BaseTacotron2
from Loss import BaseLossFunction
from Datasets import BaseTacotronDataset, BaseTacotronCollate
from Logger import BaseTacotronLogger

class Combiner():
    # tacotron_model: BaseTacotron2
    # loss_function: BaseLossFunction
    # dataset: BaseTacotronDataset
    # collate_fuction: BaseTacotronCollate
    # logger: BaseTacotronLogger
    def __init__(self, model=BaseTacotron2, loss_fn=BaseLossFunction,
                 dataset=BaseTacotronDataset, collate_fn=BaseTacotronCollate,
                 logger=BaseTacotronLogger):
        self.tacotron_model = model
        self.loss_function = loss_fn
        self.dataset = dataset
        self.collate_fuction = collate_fn
        self.logger = logger

Combiners = {
    "Origin": Combiner()
}


from Model.Lip2WavTacotron2 import Lip2WavTacotron2
from Loss.Lip2WavLoss import Lip2WavLoss
from Datasets.Lip2WavDataset import Lip2WavDataset, Lip2WavCollate
from Logger.Lip2WavLogger import Lip2WavLogger

Combiners["Lip2Wav"] = Combiner(Lip2WavTacotron2, Lip2WavLoss, Lip2WavDataset, Lip2WavCollate, Lip2WavLogger)


