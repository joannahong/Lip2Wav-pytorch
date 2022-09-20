
from mediapipe.python.solution_base import SolutionBase


FACEMESH_NUM_LANDMARKS = 468
FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478
_BINARYPB_FILE_PATH = 'mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb'


class FaceMesh(SolutionBase):

  def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            'num_faces': max_num_faces,
            'with_attention': refine_landmarks,
            'use_prev_landmarks': not static_image_mode,
        },
        calculator_params={
            'facedetectionshortrangecpu__facedetectionshortrangecommon__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'facelandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['multi_face_landmarks', 'face_detections', 'face_rects_from_landmarks', 'face_rects_from_detections'])

