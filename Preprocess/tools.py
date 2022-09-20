import os


dataset_dir = "/Users/jinchenji/Developer/Datasets/Lip2Wav/chem/intervals"      # 用于存放全部的分段数据
audio_dir = os.path.join(dataset_dir, "Audios")                                 # 用于存放音频分段
video_dir = os.path.join(dataset_dir, "Videos")                                 # 用于存放视频分段

npz_dir = os.path.join(dataset_dir, "npz")
landmark_dir = os.path.join(npz_dir, "Landmark")                        # 用于landmark分段
melspec_dir = os.path.join(npz_dir, "Melspec")


def file_traverse(dir, recursion=False, condition=None):
    file_list = os.listdir(dir)
    result = []

    for f in file_list:
        fpath = os.path.join(dir, f)

        if os.path.isdir(fpath) and recursion:
            result += file_traverse(fpath, recursion, condition=condition)
            continue

        isjoin = True
        if condition != None:
            isjoin = condition(fpath)
        if isjoin:
            result.append(fpath)

    return result



class FilePath():
    def __init__(self, origin_dir, origin_ext,
                       target_dir, target_ext, deal_exists=False):


        self.origin_dir = origin_dir
        self.origin_ext = origin_ext
        self.target_dir = target_dir
        self.target_ext = target_ext
        self.deal_exists = deal_exists

    def __condition(self, path:str):
        isFile = path.split('.')[len(path.split('.')) - 1] == self.origin_ext
        isFile = os.path.basename(path)[0] != '.' and isFile
        new_path = path.replace(self.origin_dir, self.target_dir)[:-3] + self.target_ext
        isExisits = os.path.exists(new_path)

        return isFile and (self.deal_exists or not isExisits)

    def get_file_list(self):
        return file_traverse(self.origin_dir, True, self.__condition)

    def target_path(self, path:str, mkdir=True):
        result_path = path.replace(self.origin_dir, self.target_dir)
        result_path = result_path[:-len(self.origin_ext)] + self.target_ext
        dir_path = os.path.dirname(result_path)
        if os.path.exists(dir_path) == False and mkdir:
            os.mkdir(dir_path)
        return result_path
