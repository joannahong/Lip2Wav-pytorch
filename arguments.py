import argparse
import torch.nn as nn
from Hyperparameter import HyperParameter
# from Model import Models, LossFuncations
# from Dataset import DatasetsLoad
# from Combiner import Combiners
from FaceLandmarks import fmc


class Arguments():
    main_hp: HyperParameter
    main_model: nn.Module
    opeartion = None

    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser(description='')

        parser.add_argument('--hp', help='choose the HyperParameters for trainning (default: macHP)',
                            type=str, default='macHP', metavar='N')
        parser.add_argument('--model', help='choose the Model for trainning (default: None)',
                            type=str, default=None, metavar='N')

        parser.add_argument('--annotate', help='The title of the trainning (default: None)',
                            type=str, default=None, metavar='N')

        parser.add_argument('--bts', help='batch size (default: None)',
                            type=int, default=None, metavar='N')
        parser.add_argument('--epochs', help='input max epochs for training (default: None)',
                            type=int, default=None, metavar='N')
        parser.add_argument('--lr', help='input batch size for training (default: None)',
                            type=float, default=None, metavar='N')
        parser.add_argument('--ckpt-interval', help='the interval to save the checkpoint (default: 5)',
                            type=int, default=None, metavar='N')
        parser.add_argument('--fmc', help='used face mesh collection (default: Lips)',
                            type=str, default=None, metavar='N')
        parser.add_argument('--opeartion', help='choose the preprocess opeartion (default: separate_audio)',
                            type=str, default=None, metavar='N')

        args = parser.parse_args()

        self.main_hp = HyperParameter.HyperParameters[args.hp]

        if args.model == None:
            self.combiner = Combiners[self.main_hp.main_model]

        else:
            self.combiner = Combiners[args.model]

        self.main_model = self.combiner.Tacotron
        self.loss_fn = self.combiner.LossFunction
        self.dataset = self.combiner.Dataset
        self.collate_fn = self.combiner.Collate
        self.main_logger = self.combiner.Logger


        if args.annotate != None:
            self.main_hp.train_annotate = args.annotate

        if args.bts != None:
            self.main_hp.batch_size = args.bts
        if args.epochs != None:
            self.main_hp.epochs = args.epochs
        if args.lr != None:
            self.main_hp.lr = args.lr
        if args.ckpt_interval != None:
            self.main_hp.checkpoint_interval = args.ckpt_interval
        if args.fmc != None:
            self.main_hp.fmc = fmc[args.fmc]
        # self.main_hp.load_checkpoint = True
        # self.main_hp.load_ckpt_name = "Train3090_100E_May07(20-32-42)"
        # self.main_hp.load_ckpt_epoch = 100

        self.main_hp.updateParams()

        self.opeartion = args.opeartion


arguments = Arguments()