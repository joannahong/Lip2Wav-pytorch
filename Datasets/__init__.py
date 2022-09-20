
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TacotronDataset(Dataset):
    def __init__(self, hp, splits="train+test+val", excludeZero=True):
        self.hp = hp
        self.data_dir = self.hp.datasets_dir
        self.video_dir = os.path.join(self.data_dir, "intervals/Videos/")
        self.audio_dir = os.path.join(self.data_dir, "intervals/Audios/")
        self.npz_dir = os.path.join(self.data_dir, "intervals/npz/")

        self.landmark_dir = os.path.join(self.npz_dir, self.hp.landmark_dir)
        self.spec_dir = os.path.join(self.npz_dir, "Melspec")

        self.zero_set = []
        if excludeZero:
            with open(os.path.join(self.data_dir, "intervals/zero.txt")) as data_file:
                self.zero_set = [line.strip('\n') for line in data_file.readlines()]

        self.complete_set = []
        self.interval_set = []
        self.frame_set = []
        self.splits = splits.split("+")
        self._generate_set(self.splits)

    def _generate_set(self, splits):
        for split in splits:
            with open(os.path.join(self.data_dir, "{}.txt".format(split))) as data_file:
                comp_set = [line.strip('\n') for line in data_file.readlines()]
                for comp_dir in comp_set:
                    dir_path = os.path.join(self.video_dir, comp_dir)
                    for v in os.listdir(dir_path):
                        fcode = "{}/{}".format(comp_dir, v[:-4])
                        if fcode not in self.zero_set:
                            self.interval_set.append(fcode)
                            faceRegco_dir = os.path.join(self.landmark_dir, fcode)
                            base_info_npz = os.path.join(faceRegco_dir, "base_info.npz")
                            recog_frames = np.load(base_info_npz)["recog_frame"]
                            if len(recog_frames) == 0:
                                break
                            rf = 0
                            while rf < max(recog_frames):
                                if self.isCenter(recog_frames, rf):
                                    self.frame_set.append(fcode+"/{}".format(rf))
                                    rf += self.hp.crop_overlap
                                else:
                                    rf += 1
                self.complete_set += comp_set


    def __getitem__(self, index):
        fileCode = self.frame_set[index].split('/')
        interval_ID = fileCode[0]
        cut_ID = fileCode[1]
        frame_ID = fileCode[2]
        frame_path = os.path.join(self.landmark_dir, "{}/{}/{}.npz".format(interval_ID, cut_ID, frame_ID))
        window_fnames = self.get_window(frame_path)

        spec_path = os.path.join(self.spec_dir, "{}/{}.npz".format(interval_ID, cut_ID))
        melspec = np.load(spec_path)['melspec']
        melspec = self.crop_audio_window(melspec, int(frame_ID), use_mel_step_size=True)

        window = np.zeros([self.hp.cropT, 478, 3], dtype=np.float32)
        if window_fnames != None:
            for idx, fname in enumerate(window_fnames):
                frame = np.load(fname)["landmark"]
                window[idx, :, :] = frame
            window_len = len(window_fnames)
        else:
            window = None
            window_len = 0
            print("get one None")
        return window, melspec, window_len

    def load_validation(self, frame_path, spec_path):
        window_fnames = self.get_window(frame_path)


    def __len__(self):
        return len(self.frame_set)

    def isCenter(self, recog_frames, center_id):
        window_ids = range(center_id - self.hp.cropT // 2, center_id + self.hp.cropT // 2 + self.hp.cropT % 2)
        for id in window_ids:
            if id not in recog_frames:
                return False
        return True

    def get_window(self, center_frame_path):
        vidname = os.path.dirname(center_frame_path)
        center_id = self.get_frame_id(center_frame_path)
        window_ids = range(center_id - self.hp.cropT // 2, center_id + self.hp.cropT // 2 + self.hp.cropT % 2)

        window_fnames = list()
        for frame_id in window_ids:
            frame = os.path.join(vidname, '{}.npz'.format(frame_id))

            if not os.path.isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, center_frame_ID, use_mel_step_size=False):
        start_frame_id = center_frame_ID - self.hp.cropT // 2
        end_frame_id = center_frame_ID + self.hp.cropT // 2

        total_num_frames = int((spec.shape[1] * self.hp.hop_size * self.hp.fps) / self.hp.sample_rate)

        start_idx = int(spec.shape[1] * start_frame_id / float(total_num_frames))
        end_idx = int(spec.shape[1] * end_frame_id / float(total_num_frames))
        if use_mel_step_size:
            end_idx = start_idx + self.hp.mel_step_size
        return spec[:, start_idx:end_idx]

    def get_frame_id(self, frame_path):
        return int(os.path.basename(frame_path).split('.')[0])


class TacotronCollate():
    def __init__(self, hp):
        self.hp = hp


    def __call__(self, batch):
        landmarks, spectrums, video_length = [], [], []

        for lm, sp, vl in batch:
            landmarks.append(torch.Tensor(lm))
            spectrums.append(torch.Tensor(sp))
            video_length.append(vl)

        input_lengths = torch.LongTensor([vl for vl in video_length])
        batch_size = len(input_lengths)

        LandmarkPadded = torch.FloatTensor(batch_size, 3, self.hp.cropT, self.hp.fmc.landmark_count)
        LandmarkPadded.zero_()

        SpecPadded = torch.FloatTensor(batch_size, self.hp.num_mels, self.hp.mel_step_size)
        SpecPadded.zero_()

        for i in range(batch_size):
            lm = landmarks[i].permute(2, 0, 1).contiguous()
            sp = spectrums[i]

            for ids, fmc_ids in enumerate(self.hp.fmc.collection):
                LandmarkPadded[i, :, :, ids] = lm[:, :, fmc_ids]
            SpecPadded[i, :, :sp.shape[1]] = sp

        return LandmarkPadded, SpecPadded, input_lengths

