import os
import numpy as np

from FaceLandmark.get_landmark import get_mini_video_landmark
from FaceLandmark.face_rect import get_rect, normalized_to_pixel_coordinates


class crop_frame():
    def __init__(self, speaker):
        self.speaker = speaker

    def __call__(self, frame):
        if self.speaker == "chess":
            return frame[270:460, 770:1130]
        elif self.speaker == "eh":
            return frame[250:, 450:]
        elif self.speaker == "dl":
            return frame[500:, 960:]
        else:
            return frame



def export_landmark(origin_path:str, target_path, landmark_pad=1000, speaker="chess", *args, **kwargs):
    """
    导出视频信息与 Landmark 的 npz，2500 条数据大约需要 10 小时
    Args:
        video_path:
        landmark_pad:
    Returns:

    """
    crop_fn = crop_frame(speaker=speaker)
    mini_json = get_mini_video_landmark(origin_path, crop_frame_fn=crop_fn)
    detect_info = np.zeros([landmark_pad, 14])
    landmark = np.zeros([landmark_pad, 478, 3])
    recog_frame = []

    frame_width = mini_json["video_info"]["width"]
    frame_height = mini_json["video_info"]["height"]
    for vl in mini_json["video_landmark"]:
        if vl["frame_landmark"] == None:
            continue
        frame_id = vl["frame_id"]
        if frame_id >= landmark_pad:
            break
        recog_frame.append(frame_id)
        score = vl["frame_landmark"]["detect_info"]["score"]
        rect_info = vl["frame_landmark"]["detect_info"]["face_rect"]


        normal_pc = normalized_to_pixel_coordinates(rect_info["center"], frame_width, frame_height)
        p1, p2, p3, p4 = get_rect(pc=normal_pc, width=rect_info["width"] * frame_width, height=rect_info["height"] * frame_height, angle=rect_info["rotate_angle"])
        detect_info[frame_id] = [score, rect_info["width"], rect_info["height"], rect_info["rotate_angle"],
                                 rect_info["center"][0], rect_info["center"][1],
                                 p1[0], p1[1], p2[0], p2[1],
                                 p3[0], p3[1], p4[0], p4[1]]

        for idx in range(0, 478):
            landmark[frame_id][idx] = list(vl["frame_landmark"]["landmark"][idx])
    base_info = [mini_json["video_info"]["frame_count"], frame_width, frame_height]
    recog_frame = np.array(recog_frame)

    np.savez(target_path,
             landmark=landmark,         # (landmark_pad, 478, 3)
             detect_info=detect_info,   # landmark_pad * [score, width, height, angle,
                                        #                 center_x, center_y,
                                        #                 p1x, p1y, p2x, p2y,
                                        #                 p3x, p3y, p4x, p4y]
             base_info=base_info,       # [frame_count, width, height]
             recog_frame=recog_frame)


