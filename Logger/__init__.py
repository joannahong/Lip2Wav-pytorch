import os, datetime
import logging
import numpy as np
import torch.cuda
from tensorboardX import SummaryWriter

from Hyperparameter import HyperParameter
from utils import to_arr
from utils.audio import inv_mel_spectrogram
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from matplotlib import pyplot as plt
from Metrics import STOI, PESQ


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

class makeStr:
    def __init__(self, initStr=''):
        self.string = initStr

    def addLine(self, newline):
        self.string += newline + '\n'

    def __str__(self):
        return self.string

def make_start_info(hp, device, annotate="testHP"):
    """
    在训练开始时调用，记录训练的数据（超参数数据），对这个函数的修改要同时修改 testHP.py
    :param hp: 超参数信息
    """

    strInfo = makeStr(initStr='\n')

    strInfo.addLine("=" * 50)
    strInfo.addLine("Start Train: {}".format(annotate))
    strInfo.addLine("Train On : {}".format(device))

    strInfo.addLine("=" * 50)
    strInfo.addLine("=====          Training Device Info          =====")
    strInfo.addLine("=" * 50)
    if device != 'cpu':
        properties = torch.cuda.get_device_properties(device)
        strInfo.addLine("Device Name: {}".format(properties.name))
        strInfo.addLine("Device Total Memory: {}".format(properties.total_memory))
        strInfo.addLine("Device Capability: major-{}, minor-{}".format(properties.major, properties.minor))
        strInfo.addLine("Device Multi Processor Count: {}".format(properties.multi_processor_count))
    else:
        strInfo.addLine("!!!!!             CUDA NO USE                !!!!!")
    strInfo.addLine("=" * 50)
    strInfo.addLine("Dataset Dir is: {}".format(hp.datasets_dir))
    strInfo.addLine("Use Landmark Dir: {}".format(hp.landmark_dir))
    strInfo.addLine("Dataset crop overlap = {}".format(hp.crop_overlap))
    strInfo.addLine("Use Params are: {}".format(hp.hp_name))

    strInfo.addLine("=" * 50)
    strInfo.addLine("=====          Hyperparameters Info          =====")
    strInfo.addLine("=" * 50)
    strInfo.addLine("Use Model = {}".format(hp.main_model))
    strInfo.addLine("Use Face Mesh Collection = {}".format(hp.fmc.name))
    strInfo.addLine("Batch Size = {}".format(hp.batch_size))
    strInfo.addLine("Epochs = {}".format(hp.epochs))
    strInfo.addLine("Learning Rate = {}".format(hp.lr))
    strInfo.addLine("Seed = {}".format(hp.seed))
    strInfo.addLine("Decoder Frames Step = {}".format(hp.n_frames_per_step))

    strInfo.addLine("=" * 50)
    strInfo.addLine("=====           Teacher Force Info           =====")
    strInfo.addLine("=" * 50)
    strInfo.addLine("Decay Method = {}".format(hp.main_decayer_params["decay_method_str"]))
    if hp.main_decayer_params["decay_method_str"] != 'none':
        strInfo.addLine("Init Ratio = {}".format(hp.main_decayer_params["init_ratio"]))
        strInfo.addLine("Start Decay = {}".format(hp.main_decayer_params["start_decay"]))
        strInfo.addLine("Decay Steps = {}".format(hp.main_decayer_params["decay_steps"]))
        strInfo.addLine("Decay Alpha = {}".format(hp.main_decayer_params["alpha"]))

    strInfo.addLine("=" * 50)
    strInfo.addLine("=====            CheckPoint Info             =====")
    strInfo.addLine("=" * 50)
    if hp.save_checkpoint:
        strInfo.addLine("Save ckpt interval = {}".format(hp.checkpoint_interval))
    else:
        strInfo.addLine("!!!!!    Alert, Don't Save ckpt this time    !!!!!")
    if hp.load_checkpoint:
        strInfo.addLine("Load ckpt at {}".format(hp.load_checkpoint_path))

    strInfo.addLine("=" * 50)
    return strInfo.string


class TacotronLogger(SummaryWriter):
    def __init__(self, hp:HyperParameter):

        self.hp = hp
        self.annotate = hp.timming_annotate
        self.log_dir = hp.log_dir

        super(TacotronLogger, self).__init__(self.log_dir, flush_secs=5)
        self.mainLog = os.path.join(self.log_dir, "mainLog.txt")
        self.mainLogger = get_logger(self.mainLog)

    def info(self, info):
        self.mainLogger.info(info)

    def log_start_info(self, device):
        """
        在训练开始时调用，记录训练的数据（超参数数据），对这个函数的修改要同时修改 testHP.py
        :param hp: 超参数信息
        """
        log_str = make_start_info(self.hp, device, annotate=self.annotate)
        self.mainLogger.info(log_str)

    def log_end_info(self):
        """
        在训练结束时调用，记录训练的时间信息
        """
        current_time = datetime.datetime.now().strftime('%b%d(%H-%M-%S)')
        self.mainLogger.info("="*50)
        self.mainLogger.info("Train End: {}".format(self.annotate))
        self.mainLogger.info("End Time : {}".format(current_time))

        self.close()




    def log_study_info(self, grad_norm, learning_rate, epoch_idx):
        self.info("grad_norm = {:.3f}".format(grad_norm))
        self.add_scalar("train.grad_norm", grad_norm, epoch_idx)
        self.add_scalar("train.learning_rate", learning_rate, epoch_idx)


    def log_model_output(self, predict, target, epoch_idx,
                         statge="train", save_alignment_image=False,
                         log_image=True,
                         log_audio=True,
                         log_pred_metric=False):
        """
        每个 epoch 的 eval 阶段最后执行，记录合成音频的质量
        Args:
            mel_pred: 模型预测的mel频谱
            mel_postnet: 经过 postnet 的mel频谱
            alignment:
            mel_target: 预测目标
            epoch_idx: 轮数
        """

        spec_outputs, spec_outputs_postnet, alignments = predict
        spec_pred = spec_outputs[0]
        spec_postnet = spec_outputs_postnet[0]
        spec_target = target[0]
        alignment = alignments[0]

        spec_pred = to_arr(spec_pred)
        spec_postnet = to_arr(spec_postnet)
        alignment = to_arr(alignment).T
        spec_target = to_arr(spec_target)

        # plot alignment, mel and postnet output
        if save_alignment_image:
            alimg = plot_alignment_to_numpy(alignment)
            plt.imsave(os.path.join(self.logdir, "align_{}.png".format(epoch_idx)), alimg.transpose(1, 2, 0))

        if log_image:
            self.add_image("{}.alignment".format(statge),
                           img_tensor=plot_alignment_to_numpy(alignment),
                           global_step=epoch_idx)
            self.add_image("{}.mel_outputs".format(statge),
                           img_tensor=plot_spectrogram_to_numpy(spec_pred),
                           global_step=epoch_idx)
            self.add_image("{}.mel_outputs_postnet".format(statge),
                           img_tensor=plot_spectrogram_to_numpy(spec_postnet),
                           global_step=epoch_idx)
            self.add_image("{}.mel_target".format(statge),
                           img_tensor=plot_spectrogram_to_numpy(spec_target),
                           global_step=epoch_idx)


        # save audio
        # try: # sometimes error
        wav = inv_mel_spectrogram(spec_pred, self.hp)
        wav *= 1 / max(0.01, np.max(np.abs(wav)))
        wav_postnet = inv_mel_spectrogram(spec_postnet, self.hp)
        wav_postnet *= 1 / max(0.01, np.max(np.abs(wav_postnet)))
        wav_target = inv_mel_spectrogram(spec_target, self.hp)
        wav_target *= 1 / max(0.01, np.max(np.abs(wav_target)))

        if log_audio:
            self.add_audio('{}.pred'.format(statge),
                           snd_tensor=wav,
                           global_step=epoch_idx, sample_rate=self.hp.sample_rate)
            self.add_audio('{}.pred_postnet'.format(statge),
                           snd_tensor=wav_postnet,
                           global_step=epoch_idx, sample_rate=self.hp.sample_rate)
            self.add_audio('{}.target'.format(statge),
                           snd_tensor=wav_target,
                           global_step=epoch_idx, sample_rate=self.hp.sample_rate)


        if log_pred_metric:
            stoi = STOI(wav, wav_target, self.hp.sample_rate, extended=False)
            estoi = STOI(wav, wav_target, self.hp.sample_rate, extended=True)
            pesq = PESQ(wav, wav_target, self.hp.sample_rate)

            post_stoi = STOI(wav_postnet, wav_target, self.hp.sample_rate, extended=False)
            post_estoi = STOI(wav_postnet, wav_target, self.hp.sample_rate, extended=True)
            post_pesq = PESQ(wav_postnet, wav_target, self.hp.sample_rate)

            self.add_scalars("{}.STOI".format(statge), {'pred': stoi,
                                                        'postnet': post_stoi}, epoch_idx)
            self.add_scalars("{}.ESTOI".format(statge), {'pred': estoi,
                                                        'postnet': post_estoi}, epoch_idx)
            self.add_scalars("{}.PESQ".format(statge), {'pred': pesq,
                                                        'postnet': post_pesq}, epoch_idx)


    def log_test_metrics(self, metrics, epoch_idx):
        metrics_mean = []
        for i in range(metrics.shape[1]):
            metrics_mean.append(metrics[:, i].mean())


        self.add_scalars("test.STOI", {'pred': metrics_mean[0],
                                       'postnet': metrics_mean[3]}, epoch_idx)
        self.add_scalars("test.ESTOI", {'pred': metrics_mean[1],
                                        'postnet': metrics_mean[4]}, epoch_idx)
        self.add_scalars("test.PESQ", {'pred': metrics_mean[2],
                                       'postnet': metrics_mean[5]}, epoch_idx)

        return metrics_mean
