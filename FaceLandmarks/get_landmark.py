import os, cv2
import FaceLandmarks.face_mesh_collections as fmc
from FaceLandmarks.FaceMesh import FaceMesh

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


# For static images:
def get_full_image_landmark(image, mesh_collections=[fmc.FMesh_Tesselation]):

    with FaceMesh(static_image_mode=True, max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.5) as face_recognition:
        results = face_recognition.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return

        facials_recg = []
        for idx, landmark_list in enumerate(results.multi_face_landmarks):
            face_detcet = results.face_detections[idx - 1]
            facial = {
                "facial_id": face_detcet.label_id[0],
                "detect_info": {
                    "score": face_detcet.score[0],
                    "relative_bounding_box": {
                        "xmin": face_detcet.location_data.relative_bounding_box.xmin,
                        "ymin": face_detcet.location_data.relative_bounding_box.ymin,
                        "width": face_detcet.location_data.relative_bounding_box.width,
                        "height": face_detcet.location_data.relative_bounding_box.height
                    }
                },
                "mesh_collection": []
            }

            num_landmarks = len(landmark_list.landmark)
            idx_to_coordinates = {}
            for idx, landmark in enumerate(landmark_list.landmark):
                if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
                    continue

                landmark_px = (landmark.x, landmark.y, landmark.z)
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px

            for mc in mesh_collections:
                mesh_collection = {
                    "collection_type": mc.name,
                    "landmark": []
                }

                for mesh_landmark in mc.collection:
                    if not (0 <= mesh_landmark < num_landmarks):
                        raise ValueError(f'Landmark index is out of range. Invalid lankmark '
                                         f'from landmark #{mesh_landmark}.')
                    if mesh_landmark in idx_to_coordinates:
                        landmark_point = {
                            "landmark_id": mesh_landmark,
                            "coordinates": {
                                "x": idx_to_coordinates[mesh_landmark][0] - face_detcet.location_data.relative_bounding_box.xmin,
                                "y": idx_to_coordinates[mesh_landmark][1] - face_detcet.location_data.relative_bounding_box.ymin,
                            }
                        }
                        mesh_collection["landmark"].append(landmark_point)

                facial['mesh_collection'].append(mesh_collection)
            facials_recg.append(facial)

        image_height, image_width, _ = image.shape
        image_landmark = {
            "facials_count": len(facials_recg),
            "image_info": {
                "height": image_height,
                "width": image_width
            },
            "facials_recognition": facials_recg
        }
        return image_landmark

# For video:
def get_full_video_landmark(video_file_path, mesh_collections=[fmc.FMesh_Tesselation], crop_frame_fn=None):
    cap = cv2.VideoCapture(video_file_path)
    cap.isOpened()
    frame_count = 0
    success = True
    landmark_infos = []
    while (success):
        success, frame = cap.read()
        if not success:
            break
        if crop_frame_fn != None:
            frame = crop_frame_fn(frame)
        landmark = get_full_image_landmark(frame, mesh_collections=mesh_collections)
        landmark_info = {
            "frame_id": frame_count,
            "frame_landmark": landmark
        }
        landmark_infos.append(landmark_info)
        frame_count += (1 if success else 0)

    video_lanmark = {
        "video_name": os.path.basename(video_file_path),
        "frame_count": frame_count,
        "video_landmark": landmark_infos
    }

    cap.release()
    return video_lanmark


# For static images (mini json):
def get_mini_image_landmark(image):

    with FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_recognition:
        results = face_recognition.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        face_detcet = results.face_detections[0]
        landmark_list = results.multi_face_landmarks[0]

        face_rects = results.face_rects_from_landmarks[0]
        # face_rects = results.face_rects_from_detections[0]

        pc = (face_rects.x_center, face_rects.y_center)
        w, h = face_rects.width, face_rects.height
        rot_angle = face_rects.rotation

        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (
                    landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
                continue

            landmark_px = (landmark.x, landmark.y, landmark.z)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

        image_height, image_width, _ = image.shape



        image_landmark = {
            "detect_info": {
                "score": face_detcet.score[0],
                # "relative_bounding_box": {
                #     "xmin": face_detcet.location_data.relative_bounding_box.xmin,
                #     "ymin": face_detcet.location_data.relative_bounding_box.ymin,
                #     "height": face_detcet.location_data.relative_bounding_box.height,
                #     "width": face_detcet.location_data.relative_bounding_box.width
                # },
                "face_rect": {
                    "center": pc,
                    "width": w,
                    "height": h,
                    "rotate_angle": rot_angle,
                }
            },
            "landmark": idx_to_coordinates
        }

        return image_landmark

# For video (mini json):
def get_mini_video_landmark(video_file_path, crop_frame_fn=None):
    cap = cv2.VideoCapture(video_file_path)
    cap.isOpened()
    frame_count = 0
    success = True
    landmark_infos = []
    while (success):
        success, frame = cap.read()
        if not success:
            break
        if crop_frame_fn != None:
            frame = crop_frame_fn(frame)
        landmark = get_mini_image_landmark(frame)
        landmark_info = {
            "frame_id": frame_count,
            "frame_landmark": landmark
        }
        landmark_infos.append(landmark_info)
        frame_count += (1 if success else 0)

    video_lanmark = {
        "video_info": {
            "video_name": os.path.basename(video_file_path),
            "frame_count": frame_count,
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        },
        "video_landmark": landmark_infos
    }

    cap.release()
    return video_lanmark

