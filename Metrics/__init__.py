
import pysepm
from pystoi import stoi
from pesq import pesq
from pesq.cypesq import NoUtterancesError



def STOI(audio, target, sr, extended=False):
    """
    (扩展)短时客观可理解性
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率
        extended: 为 True 时计算 ESTOI

    Returns: 0-1 . 越高越好.
    """
    return stoi(target, audio, sr, extended)

def PESQ(audio, target, sr, mode='wb'):
    """
    语音质量评估
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率，仅接受 8000hz 或 16000hz
        mode: 'wb' (wide-band) or 'nb' (narrow-band, sr=8000)

    Returns: -0.5 - 4.5 . 越高越好.
    """
    try:
        result = pesq(sr, target, audio, mode)
    except NoUtterancesError:
        return -0.5
    return result



def SNRseg(audio, target, sr):
    """
    分段信噪比
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率

    Returns: 越高越好.
    """
    return pysepm.SNRseg(target, audio, sr)

def LLR(audio, target, sr):
    """
    对数似然比测度
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率
    Returns: 越高越好.
    """
    return pysepm.llr(target, audio, sr)

def WSS(audio, target, sr):
    """
    加权谱倾斜测度
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率
    Returns: 越小越好.
    """
    return pysepm.wss(target, audio, sr)

def CepstrumDistance(audio, target, sr):
    """
    倒谱距离
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率
    Returns: 越高越好.
    """
    return pysepm.cepstrum_distance(target, audio, sr)

def Composite(audio, target, sr):
    """
    综合
    Args:
        audio: 计算的音频
        target: 计算的对象
        sr: 采样率
    Returns: CSIG , CBAK , COVL all from 1 - 5 , 越高越好.
    """
    CSIG, CBAK, COVL = pysepm.composite(target, audio, sr)
    return CSIG, CBAK, COVL