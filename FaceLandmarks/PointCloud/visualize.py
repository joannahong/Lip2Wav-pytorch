import os
import cv2
import numpy as np
import FaceLandmarks.face_mesh_collections as fmc
from FaceLandmarks.PointCloud.draw import *
from FaceLandmarks.PointCloud.transformation import *


def select_color(idx):
    c = "black"
    if idx in fmc.FMesh_Features_Contours.collection:
        c = 'magenta'
    elif idx in fmc.FMesh_Lips.collection:
        c = 'light-red'
    elif idx in fmc.FMesh_Lips_Content.collection:
        c = 'light-yellow'
    elif idx in fmc.FMesh_Face_Edge.collection:
        c = 'light-green'
    elif idx in fmc.FMesh_Forehead.collection:
        c = 'light-blue'
    return c


class Landmarks():
    def __init__(self, landmarks:np.ndarray):
        self.landmarks = landmarks      # 1000, 478, 3
        self.h = 720
        self.w = 1280

    def isNone(self):
        return self.landmarks.sum() == 0

    def export_clp_video(self, video_path, fmc=fmc.FMesh_Whole_Face):
        """
        显示点云动画

            landmarks: 1000, 784, 3
        Returns:

        """
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(video_path, fourcc, 15, (self.w, self.h))

        for i in range(len(self.landmarks)):
            img = faceLandmark(self.landmarks[i, :, :].squeeze()).landmark_img(fmc)
            writer.write(img)

    def normalize_all_face(self):
        newLandmarks = np.zeros_like(self.landmarks)
        for i in range(newLandmarks.shape[0]):
            landmark = self.landmarks[i, :, :].squeeze()
            landmark = faceLandmark(landmark)
            newLandmarks[i, :, :] = landmark.normalize_face()
        return newLandmarks




class faceLandmark():
    def __init__(self, landmark):
        self.landmark = landmark      # 478, 3
        self.w = 1280
        self.h = 720

    def isNone(self):
        return self.landmark.sum() == 0

    def landmark_img(self, fmc=fmc.FMesh_Whole_Face, color=None):
        img = (np.zeros([720, 1280, 3]) + 255).astype(np.uint8)
        for f in fmc.collection:
            point = self.landmark[f]
            p = normalized_to_pixel_coordinates(point, self.w, self.h)
            c = color if color != None else select_color(f)
            drawPoint(img, p, 1, c=c)
        return img

    def mark_all_clp(self, video_path, fmc=fmc.FMesh_Whole_Face):
        """
        显示点云依序出现的动画
        """
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (self.w, self.h))
        img = (np.zeros([self.h, self.w, 3]) + 255).astype(np.uint8)
        for i in fmc.collection:
            point = self.landmark[i]
            p = normalized_to_pixel_coordinates(point, self.w, self.h)
            c = select_color(i)
            drawPoint(img, p, 1, c=c)
            writer.write(img)

    def mark_depth(self, video_path):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (self.w, self.h))

        img = (np.zeros([720, 1280, 3]) + 255).astype(np.uint8)
        z_axis = self.landmark[:, 2]
        depth, depth_max = min(z_axis), max(z_axis)
        step = (depth_max - depth) / 100
        while depth <= depth_max + 2 * step:
            for i in range(0, 478):
                point = self.landmark[i]
                if point[2] <= depth:
                    p = normalized_to_pixel_coordinates(point, self.w, self.h)
                    c = select_color(i)
                    drawPoint(img, p, 1, c=c)
            writer.write(img)
            depth += step

    def normalize_face(self):
        face_landmark = self.landmark
        eps = np.finfo(np.float32).eps

        face_landmark = np.insert(face_landmark, face_landmark.shape[1], values=np.ones(face_landmark.shape[0]), axis=1)
        face_landmark = face_landmark.T

        up, lp, dp, rp = keyPoints(face_landmark)
        center = (lp + rp) / 2

        xVec = line_direction_vector(lp, rp)
        HLine = line_direction_vector(dp, up)
        zVec = get_orth_vec(xVec, HLine)

        alpth = np.arctan(zVec[1] / (zVec[2] + eps))
        beta = np.arctan(zVec[0] / (np.sqrt(zVec[1] ** 2 + zVec[2] ** 2) + eps))

        transMat = np.dot(rotMat(alpth, 'x'), shiftMat(-center[0], -center[1], -center[2]))
        transMat = np.dot(rotMat(beta, 'y'), transMat)
        face_landmark = np.dot(transMat, face_landmark)

        up, lp, dp, rp = keyPoints(face_landmark)
        xVec = line_direction_vector(lp, rp)
        gamma = np.arctan(-xVec[1] / (xVec[0] + eps))
        face_landmark = np.dot(rotMat(gamma, 'z'), face_landmark)

        up, lp, dp, rp = keyPoints(face_landmark)
        sx, sy = 0.1 / (distance(lp, rp) + eps), 0.25 / (distance(up, dp) + eps)
        transMat = np.dot(shiftMat(0.5, 0.5, 0), scalMat(sx, sy, sz=1))
        face_landmark = np.dot(transMat, face_landmark)

        face_landmark = face_landmark.T
        return face_landmark[:, :3]

    def rotation(self, dx=0, dy=0, dz=0):
        face_landmark = np.insert(self.landmark, self.landmark.shape[1], values=np.ones(self.landmark.shape[0]), axis=1)
        face_landmark = face_landmark.T

        rotx = rotMat(math.radians(dx), 'x')
        roty = rotMat(math.radians(dy), 'y')
        rotz = rotMat(math.radians(dz), 'z')

        face_landmark = np.dot(rotx, face_landmark)
        face_landmark = np.dot(roty, face_landmark)
        face_landmark = np.dot(rotz, face_landmark)

        face_landmark = face_landmark.T
        return face_landmark[:, :3]

    def shift(self, dx, dy, dz):
        face_landmark = np.insert(self.landmark, self.landmark.shape[1], values=np.ones(self.landmark.shape[0]), axis=1)
        face_landmark = face_landmark.T

        face_landmark = np.dot(shiftMat(dx, dy, dz), face_landmark)

        face_landmark = face_landmark.T
        return face_landmark[:, :3]

    def scale(self, sx, sy, sz):
        face_landmark = np.insert(self.landmark, self.landmark.shape[1], values=np.ones(self.landmark.shape[0]), axis=1)
        face_landmark = face_landmark.T

        face_landmark = np.dot(scalMat(sx,sy,sz), face_landmark)

        face_landmark = face_landmark.T
        return face_landmark[:, :3]

    def compare_img(self, other, show=True):

        self_img = self.landmark_img()
        other_img = other.landmark_img()

        show_img = cv2.hconcat([self_img, other_img])  # 水平拼接
        if show:
            cv2.imshow("show", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return show_img
