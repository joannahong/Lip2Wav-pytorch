import os
import numpy as np
from tqdm import tqdm
from Preprocess.tools import *
from FaceLandmark.get_landmark import get_mini_video_landmark
from FaceLandmark.face_rect import get_rect, normalized_to_pixel_coordinates


interval_dir = "/home/g20tka12/Developer/LipDatasetTools/Datasets/chem/intervals"
origin_npz_dir = os.path.join(interval_dir, "npz/Landmark")
new_npz_dir = os.path.join(interval_dir, "npz/faceRect")
video_dir = os.path.join(interval_dir, "Videos")


def addRect(video_path:str, landmark_pad:int):
    npz_path = vf.replace(video_dir, origin_npz_dir)[:-4] + ".npz"
    new_npz_path = npz_path.replace(origin_npz_dir, new_npz_dir)

    mini_json = get_mini_video_landmark(video_path)
    detect_info = np.zeros([landmark_pad, 14])

    width = mini_json["video_info"]["width"]
    height = mini_json["video_info"]["height"]
    for vl in mini_json["video_landmark"]:
        if vl["frame_landmark"] == None:
            continue
        frame_id = vl["frame_id"]
        score = vl["frame_landmark"]["detect_info"]["score"]
        rect_info = vl["frame_landmark"]["detect_info"]["face_rect"]

        normal_pc = normalized_to_pixel_coordinates(rect_info["center"], width, height)
        p1, p2, p3, p4 = get_rect(pc=normal_pc, width=rect_info["width"] * width, height=rect_info["height"] * height,
                                  angle=rect_info["rotate_angle"])

        detect_info[frame_id] = [score, rect_info["width"], rect_info["height"], rect_info["rotate_angle"],
                                 rect_info["center"][0], rect_info["center"][1],
                                 p1[0], p1[1], p2[0], p2[1],
                                 p3[0], p3[1], p4[0], p4[1]]

    content = np.load(npz_path)

    if not os.path.exists(os.path.dirname(new_npz_path)):
        os.mkdir(os.path.dirname(new_npz_path))

    np.savez(new_npz_path,
             landmark=content["landmark"],
             detect_info=detect_info,
             base_info=content["base_info"])


if __name__ == '__main__':

    # 导出视频信息与 Landmark 的 npz，2500 条数据大约需要 10 小时

    interval_dir = "/home/g20tka12/Developer/LipDatasetTools/Datasets/chem/intervals"
    origin_npz_dir = os.path.join(interval_dir, "npz/Landmark")
    new_npz_dir = os.path.join(interval_dir, "npz/faceRect")
    video_dir = os.path.join(interval_dir, "Videos")

    def showVideo(path: str) -> bool:
        isVideoFile = path.split('.')[len(path.split('.')) - 1] == 'mp4'
        isVideoFile = os.path.basename(path)[0] != '.' and isVideoFile
        new_path = path.replace(video_dir, new_npz_dir)[:-4] + ".npz"
        isExisits = os.path.exists(new_path)

        return isVideoFile and not isExisits


    video_files = file_traverse(video_dir, recursion=True, condition=showVideo)
    for vf in tqdm(video_files):
        addRect(vf, 1000)
