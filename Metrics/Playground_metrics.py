

from Metrics import STOI, PESQ
import pysepm
from utils.audio import *
from arguments import arguments as arg

hp = arg.main_hp

iter = 65000
postnet = "/Users/jinchenji/Developer/JetBrains/Pycharm/LipLandmark2Wav/tmp/postnet_{}.wav".format(iter)
target = "/Users/jinchenji/Developer/JetBrains/Pycharm/LipLandmark2Wav/tmp/target_{}.wav".format(iter)


postnet = load_wav(postnet, hp)
target = load_wav(target, hp)
save_wav(target, '/Users/jinchenji/Downloads/target.wav', hp)
print(len(target))


print(type(target))


# Method 2: llr (对数似然比测度)
        #
        # The higher the score, the better the performance.
llr = pysepm.llr(target, postnet, hp.sample_rate)
print("llr: ", llr)


# Method 3: WSS (加权谱倾斜测度)
        #
        # The smaller the score, the better the performance.
# wss = pysepm.wss(ref, deg, sr0)
# print("wss: ", wss)


# Method 4: STOI (可短时客观可懂)
        # the score from 0-1 . The higher the score, the better the performance.
sstoi = STOI(postnet, target, hp.sample_rate)
print("sstoi: ", sstoi)
estoi = STOI(postnet, target, hp.sample_rate, extended=True)
print("estoi: ", estoi)



# Method 5: PESQ
        # The score from -0.5 - 4.5 .The higher the score, the better the performance.
pesq = PESQ(postnet, target, hp.sample_rate, mode='nb')
print("pesq: ", pesq)


# Method 6: CD (Cepstrum Distance)
        #
        # The higher the score, the better the performance.
# cepstrum_distance = pysepm.cepstrum_distance(ref, deg, sr0)
# print("ceps: ", cepstrum_distance)


# Method 7: LSD (对数谱距离)
        # This method I use the LSD.py to calculate the distance
        # The smaller the score, the better the performance.



# Method 8: Composite
        # In this method , It comes some errors, if you want to solve the error ,  see the step 8 in this file.
        # CSIG , CBAK , COVL all from 1 - 5 , The higher the score, the better the performance.
# CSIG, CBAK, COVL = pysepm.composite(ref, deg, sr0)
# CSIGs.append(CSIG)
# CBAKs.append(CBAK)
# COVLs.append(COVL)