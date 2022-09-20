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
    if idx in fmc.FMesh_Lips.collection:
        c = 'light-red'
    if idx in fmc.FMesh_Lips_Content.collection:
        c = 'light-yellow'
    if idx in fmc.FMesh_Face_Edge.collection:
        c = 'light-green'
    if idx in fmc.FMesh_Forehead.collection:
        c = 'light-blue'
    return c


class LandmarkSeries():
    def __init__(self, landmarks: np.ndarray):
        self.landmarks = landmarks  # 1000, 478, 3
        self.h = 720
        self.w = 1280
        self.fps = 30

    def isNone(self):
        return self.landmarks.sum() == 0

    def export_clp_video(self, video_path, fmc=fmc.FMesh_Whole_Face):
        """
        显示点云动画

            landmarks: 1000, 784, 3
        Returns:

        """
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.w, self.h))

        for i in range(len(self.landmarks)):
            img = faceLandmark(self.landmarks[i, :, :].squeeze()).landmark_img(fmc)
            writer.write(img)

    def normalize_all_face(self, norm_method="norm1"):
        newLandmarks = np.zeros_like(self.landmarks)
        for i in range(newLandmarks.shape[0]):
            landmark = self.landmarks[i, :, :].squeeze()
            if landmark.sum() == 0:
                newLandmarks[i, :, :] = 0
                continue
            landmark = faceLandmark(landmark, norm_method=norm_method)
            newLandmarks[i, :, :] = landmark.norm_method().landmark
        return LandmarkSeries(newLandmarks)


class faceLandmark():
    def __init__(self, landmark, norm_method="norm1"):
        self.landmark = landmark  # 478, 3
        self.up = self.landmark[10, :]      # 面部
        self.lp = self.landmark[93, :]
        self.dp = self.landmark[152, :]
        self.rp = self.landmark[323, :]

        # self.up = self.landmark[0, :]     # 嘴唇
        # self.lp = self.landmark[61, :]
        # self.dp = self.landmark[17, :]
        # self.rp = self.landmark[291, :]
        self.center = (self.lp + self.rp) / 2

        self.w = 1280
        self.h = 720
        self.fps = 30
        norm_methods = {
            "norm1": self.normalize_face1,
            "norm2": self.normalize_face2
        }
        self.norm_method = norm_methods[norm_method]

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
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.w, self.h))
        img = (np.zeros([self.h, self.w, 3]) + 255).astype(np.uint8)
        for i in fmc.collection:
            point = self.landmark[i]
            p = normalized_to_pixel_coordinates(point, self.w, self.h)
            c = select_color(i)
            drawPoint(img, p, 1, c=c)
            writer.write(img)

    def mark_depth(self, video_path):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.w, self.h))

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

    def compare_img(self, other, show=True):

        self_img = self.landmark_img()
        other_img = other.landmark_img()

        show_img = cv2.hconcat([self_img, other_img])  # 水平拼接
        if show:
            cv2.imshow("show", show_img)
            cv2.moveWindow("show", 200, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return show_img


    # def get_angel(self):


    def shift(self, dx=0, dy=0, dz=0, vec=None):
        if type(vec) == np.ndarray:
            return faceLandmark(self.landmark + vec)
        return faceLandmark(self.landmark + [dx, dy, dz])

    def rotation(self, dx=0, dy=0, dz=0, shift=True):
        face_landmark = self.landmark.T
        if shift:
            face_landmark = self.landmark - self.center
            face_landmark = face_landmark.T

        rotx = rotMat(dx, 'x')
        roty = rotMat(dy, 'y')
        rotz = rotMat(dz, 'z')

        face_landmark = np.dot(rotx, face_landmark)
        face_landmark = np.dot(roty, face_landmark)
        face_landmark = np.dot(rotz, face_landmark)

        face_landmark = face_landmark.T
        if shift:
            face_landmark = face_landmark + self.center

        return faceLandmark(face_landmark)

    def scale(self, sx=1, sy=1, sz=1, shift=True):
        face_landmark = self.landmark.T
        if shift:
            face_landmark = self.landmark - self.center
            face_landmark = face_landmark.T

        face_landmark = np.dot(scalMat(sx, sy, sz), face_landmark)

        face_landmark = face_landmark.T
        if shift:
            face_landmark = face_landmark + self.center
        return faceLandmark(face_landmark)


    def normalize_face1(self):
        """
        包含旋转、平移至画面中心、缩放至一定比例等操作
        """
        eps = np.finfo(np.float32).eps

        xVec = line_direction_vector(self.lp, self.rp)
        HLine = line_direction_vector(self.dp, self.up)
        zVec = get_orth_vec(xVec, HLine)

        alpth = np.arctan(zVec[1] / (zVec[2] + eps))
        beta = np.arctan(zVec[0] / (np.sqrt(zVec[1] ** 2 + zVec[2] ** 2) + eps))
        face_landmark = self.rotation(dx=alpth, dy=beta)

        xVec = line_direction_vector(face_landmark.lp, face_landmark.rp)
        gamma = np.arctan(-xVec[1] / (xVec[0] + eps))
        face_landmark = face_landmark.rotation(dz=gamma)
        #
        sx = 0.100 / (distance(face_landmark.lp, face_landmark.rp) + eps)       # 全脸设置为 0.100，嘴唇设置为 0.035
        sy = 0.250 / (distance(face_landmark.up, face_landmark.dp) + eps)
        face_landmark = face_landmark.scale(sx, sy)


        vec = np.array([0.5, 0.5, 0]) - face_landmark.center
        face_landmark = face_landmark.shift(vec=vec)

        return face_landmark

    def normalize_face2(self):
        """
        仅平移并缩放至一定比例，不对面部进行旋转
        Returns:
        """
        eps = np.finfo(np.float32).eps

        #
        # sx = 0.100 / (distance(self.lp, self.rp) + eps)       # 全脸设置为 0.100，嘴唇设置为 0.035
        # sy = 0.250 / (distance(self.up, self.dp) + eps)
        # face_landmark = self.scale(sx, sy)

        vec = np.array([0.5, 0.5, 0]) - self.center
        face_landmark = self.shift(vec=vec)

        return face_landmark





    # def normalize_face_old(self):
    #     face_landmark = self.landmark
    #     eps = np.finfo(np.float32).eps
    #
    #     face_landmark = np.insert(face_landmark, face_landmark.shape[1], values=np.ones(face_landmark.shape[0]), axis=1)
    #     face_landmark = face_landmark.T
    #
    #     up, lp, dp, rp = keyPoints(face_landmark)
    #     center = (lp + rp) / 2
    #
    #     xVec = line_direction_vector(lp, rp)
    #     HLine = line_direction_vector(dp, up)
    #     zVec = get_orth_vec(xVec, HLine)
    #
    #     alpth = np.arctan(zVec[1] / (zVec[2] + eps))
    #     beta = np.arctan(zVec[0] / (np.sqrt(zVec[1] ** 2 + zVec[2] ** 2) + eps))
    #
    #     transMat = np.dot(rotMat(alpth, 'x'), shiftMat(-center[0], -center[1], -center[2]))
    #     transMat = np.dot(rotMat(beta, 'y'), transMat)
    #     face_landmark = np.dot(transMat, face_landmark)
    #
    #     up, lp, dp, rp = keyPoints(face_landmark)
    #     xVec = line_direction_vector(lp, rp)
    #     gamma = np.arctan(-xVec[1] / (xVec[0] + eps))
    #     face_landmark = np.dot(rotMat(gamma, 'z'), face_landmark)
    #
    #     up, lp, dp, rp = keyPoints(face_landmark)
    #     sx, sy = 0.1 / (distance(lp, rp) + eps), 0.25 / (distance(up, dp) + eps)
    #     transMat = np.dot(shiftMat(0.5, 0.5, 0), scalMat(sx, sy, sz=1))
    #     face_landmark = np.dot(transMat, face_landmark)
    #
    #     face_landmark = face_landmark.T
    #     return face_landmark[:, :3]






