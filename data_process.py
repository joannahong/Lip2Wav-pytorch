import os
from glob import glob
filelist = []

with open('/mnt/hard2/joannahong/lipreading/Lip2Wav-master/Dataset/sjp/test.txt') as vidlist:
    for vid_id in vidlist:
        vid_id = vid_id.strip()
        filelist.extend(list(glob(os.path.join('/mnt/hard2/joannahong/lipreading/Lip2Wav-master/Dataset/sjp', 'preprocessed', vid_id, '*/*.jpg'))))
        # filelist.extend(list(glob(os.path.join('/mnt/hard2/joannahong/lipreading/Lip2Wav-master/Dataset/sjp', 'preprocessed_new', vid_id, '*/*.jpg'))))

print(filelist)