import numpy as np
import FaceLandmark.PointCloud as pcl


origin = "/Users/jinchenji/Developer/Datasets/Lip2Wav/chem/intervals/npz/Landmark"
target = "/Users/jinchenji/Developer/Datasets/Lip2Wav/chem/intervals/npz/NormLandmark"


def norm_face(origin_path, target_path, *args, **kwargs):
    landmarks = np.load(origin_path)["landmark"]
    newLandmarks = pcl.LandmarkSeries(landmarks).normalize_all_face()

    np.savez(target_path,
             landmark=newLandmarks.landmarks,
             detect_info=np.load(origin_path)["detect_info"],
             base_info=np.load(origin_path)["base_info"])

