import os
import numpy as np

def save_recog_face(origin_path, target_path, *args, **kwargs):
    landmarks = np.load(origin_path)["landmark"]
    detect_infos = np.load(origin_path)["detect_info"]
    base_infos = np.load(origin_path)["base_info"]

    target_dir = target_path[:-4]
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    recog_frame = []
    for idx in range(landmarks.shape[0]):
        landmark = landmarks[idx, :, :]
        if landmark.sum() != 0:

            save_path = os.path.join(target_dir, "{}.npz".format(idx))
            np.savez(save_path,
                     landmark=landmark,
                     detect_info=detect_infos[idx, :],
                     frame_idx=idx)
            recog_frame.append(idx)


    np.savez(os.path.join(target_dir, "base_info.npz"),
             landmark=base_infos,
             recog_frame=recog_frame,
             num_racog_frame=len(recog_frame))
