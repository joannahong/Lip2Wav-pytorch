
import numpy as np
import FaceLandmark.PointCloud as pcl




def padding_mel(origin_path, target_path):
    landmarks = np.load(origin_path)["landmark"]
    newLandmarks = pcl.LandmarkSeries(landmarks).normalize_all_face()

    np.savez(target_path,
             landmark=newLandmarks,
             detect_info=np.load(target_path)["detect_info"],
             base_info=np.load(target_path)["base_info"])
