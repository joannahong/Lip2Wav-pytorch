import numpy as np
import torch, os, cv2
import torch.nn.functional as F
from arguments import arguments as arg
from utils.audio import load_wav, melspectrogram, linearspectrogram

hp = arg.main_hp

dataset_dir = os.path.join(hp.datasets_dir, "intervals")                        # 用于存放全部的分段数据
audio_dir = os.path.join(dataset_dir, "Audios")                                 # 用于存放音频分段
video_dir = os.path.join(dataset_dir, "Videos")                                 # 用于存放视频分段

npz_dir = os.path.join(dataset_dir, "npz")
landmark_dir = os.path.join(npz_dir, "Landmark")                                # 用于landmark分段
melspec_dir = os.path.join(npz_dir, "Melspec")

norm_dir = os.path.join(npz_dir, "NormLandmark")



def get_auido_mask(landmarks, video_len, audio_len, spec_len):
    wav_mask = np.zeros(audio_len)
    wavK =  audio_len / video_len
    spec_mask = np.zeros(spec_len)
    specK = spec_len / video_len

    for idx in range(landmarks.shape[0]):
        if landmarks[idx, :, :].sum() != 0:
            wav_mask[int(idx * wavK):int((idx+1) * wavK)] = 1
            spec_mask[int(idx * specK): int((idx + 1) * specK)] = 1
    return wav_mask, spec_mask


def export_specs(origin_path, target_path, hyperparam=hp, premask=False, *args, **kwargs):
    audio_wav = load_wav(origin_path, hyperparam)                     # np.array([audio_time*sr, ])

    melspec = melspectrogram(audio_wav, hyperparam)                  # np.array([num_mels, len(wav)/hop_size])

    landmark_path = origin_path.replace(audio_dir, landmark_dir)[:-4] + '.npz'
    landmarks = np.load(landmark_path)["landmark"]
    video_path = origin_path.replace(audio_dir, video_dir)[:-4] + '.mp4'
    video_len = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)

    wav_mask, spec_mask = get_auido_mask(landmarks, video_len, audio_wav.shape[0], melspec.shape[1])

    if premask:
        audio_wav = audio_wav * wav_mask
        melspec = melspectrogram(audio_wav, hyperparam)                  # np.array([num_mels, len(wav)/hop_size])
    linespec = linearspectrogram(audio_wav, hyperparam)

    np.savez(target_path,
             melspec=melspec,           # (num_mels, mel_len)
             linespec=linespec,         # (num_mels, line_len)
             spec_mask=spec_mask,
             audio=audio_wav,           # (audio_time * sr, )
             audio_mask=wav_mask)

