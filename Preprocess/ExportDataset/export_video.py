import os
import cv2

import numpy as np




def export_video(origin_path:str, target_path):
    """
    导出视频信息与 Landmark 的 npz，2500 条数据大约需要 10 小时
    Args:
        video_path:
        landmark_pad:
    Returns:

    """
    videoCapture = cv2.VideoCapture(origin_path)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    # 读帧
    full_video = []
    success = True
    while success:
        success, frame = videoCapture.read()  # 获取下一帧
        if not success:
            break
        full_video.append(np.expand_dims(frame, axis=0))
    videoCapture.release()

    full_video = np.concatenate(full_video)


    np.savez(target_path,
             video=full_video,         # (landmark_pad, 478, 3)
             base_info=np.array([frame_count, w, h, fps]))       # [frame_count, width, height, fps]

