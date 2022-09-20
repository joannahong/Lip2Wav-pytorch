import os, json
from FaceLandmark.get_landmark import get_mini_video_landmark


def export_dataset(video_path):
    mini_json = get_mini_video_landmark(video_path)

    json_path = video_path.replace('mp4', 'json')
    json_path = json_path.replace('intervals/Videos/', 'intervals/Landmarks/')
    dir_name = os.path.dirname(json_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print(json_path)
    t = json.dumps(mini_json, indent=2)
    with open(json_path, 'w') as json_file:
        json_file.write(t)



