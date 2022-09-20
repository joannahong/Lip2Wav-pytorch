

面部特征点提取参考 [mediapipe](https://google.github.io/mediapipe/)

1. 安装 mediapipe 的发行版
```
pip install mediapipe
```

2. 下载 mediapipe 的源代码, 并改名为 mediapipes
```shell
git clone https://github.com/google/mediapipe.git
```

3. 修改 `mediapipes/mediapipe/python/solutions/face_mesh.py` 第 `107` 行的内容：
```python
outputs=['multi_face_landmarks', 'face_detections', 'face_rects_from_landmarks', 'face_rects_from_detections']
```

```python
full_video_lanmark = {
    "video_name": str,			# 视频文件名称
    "frame_count": int,			# 视频帧数量
    "video_landmark": [{
        "frame_id": int,		# 视频帧编号
        "frame_landmark": {
            "facials_count": int,	# 图片中识别到的面容数量
            "image_info": {			# 图片信息
                "height": int,		# 图片高度
                "width": int 		# 图片宽度
            },
            "facials_recognition": [{		# 识别到的面容
                "facial_id": int,			# 面容编号
                "detect_info": {			# 面容检测的信息
                    "score": float, 			# 检测评分（小于1.0）
                    "relative_bounding_box": {	# 面部框信息
                        "xmin": float,			# 左下角 x（小于1.0）
                        "ymin": float,			# 左下角 y（小于1.0）
                        "width": float,			# 宽度（小于1.0）
                        "height": float			# 高度（小于1.0）
                    }
                },
                "mesh_collection": [{			# 面容点集合
                    "collection_type": str,		# 面容点集合类型
                    "landmark": [{				# 面部标注点信息
                        "landmark_id": int,		# 面部标注点编号
                        "coordinates": {		# 坐标（相对于边界框左下角点）
                            "x": float,			# 坐标 x（小于1.0）
                            "y": float			# 坐标 y（小于1.0）
                        }
                    }]
                }]
            }]
        }
    }]
}


mini_video_lanmark = {
    "video_info": {
        "video_name": str,      # 视频文件名称
        "frame_count": int,     # 视频帧数量
        "width": int,            # 视频宽度
        "height": int           # 视频高度
    },
    "video_landmark": [{
        "frame_id": int,        # 视频帧编号
        "frame_landmark": { 
            "detect_info": { 
                "score": float,             # 检测评分（小于1.0）
                "face_rect": {  # 面容识别边界框
                    "center": (float, float),       # 中心点坐标
                    "width": float,			        # 宽度（小于1.0）
                    "height": float,			    # 高度（小于1.0）
                    "rotate_angle": float           # 方框旋转角度（弧度制）
                }
            },
            "landmark": {                       # 面容点集合（并不相对与边界框）
                "0": [float, float, float],     # 面容点坐标 [x, y, z]（小于1.0）
                "1": [float, float, float], 
                # ...
                "476": [float, float, float],
                "477": [float, float, float]
            }
        }
    }] 
}
```





```shell
FACEMESH_CONTOURS		# 眼部轮廓（双侧眼睛、嘴唇、眉毛）
FACEMESH_FACE_OVAL		# 脸外部轮廓
FACEMESH_IRISES			# 双侧虹膜
FACEMESH_LEFT_EYE		# 左眼
FACEMESH_LEFT_EYEBROW	# 左侧眉毛
FACEMESH_LEFT_IRIS		# 左侧虹膜
FACEMESH_LIPS			# 嘴唇
FACEMESH_RIGHT_EYE		# 右眼
FACEMESH_RIGHT_EYEBROW	# 右侧眉毛
FACEMESH_RIGHT_IRIS		# 右侧虹膜
FACEMESH_TESSELATION 	# 脸部点位镶嵌
```

