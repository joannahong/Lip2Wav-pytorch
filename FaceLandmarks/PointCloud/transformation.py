import os, math
import numpy as np


def norm_vector(vec):
    sum = np.power(vec, 2).sum()
    if sum == 0:
        return vec
    t = np.sqrt(sum)
    return vec/t

def line_direction_vector(p1, p2):
    return norm_vector(p2-p1)

def distance(p1, p2):
    d = np.power((p2 - p1)[:2], 2).sum()
    return np.sqrt(d)

def get_orth_vec(vec1, vec2):
    return norm_vector(np.cross(vec1[:3], vec2[:3]))

def rotMat(theta, axis='x'):
    cos, sin = np.cos(theta), np.sin(theta)
    rotMat = None
    if axis=='x':
        rotMat = np.array([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos],
        ])
    elif axis=='y':
        rotMat = np.array([
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos],
        ])
    elif axis=='z':
        rotMat = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ])
    return rotMat

def rotMat_old(theta, axis='x'):
    cos, sin = np.cos(theta), np.sin(theta)
    rotMat = None
    if axis=='x':
        rotMat = np.array([
            [1, 0, 0, 0],
            [0, cos, -sin, 0],
            [0, sin, cos, 0],
            [0, 0, 0, 1]
        ])
    elif axis=='y':
        rotMat = np.array([
            [cos, 0, sin, 0],
            [0, 1, 0, 0],
            [-sin, 0, cos, 0],
            [0, 0, 0, 1]
        ])
    elif axis=='z':
        rotMat = np.array([
            [cos, -sin, 0, 0],
            [sin, cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    return rotMat

def scalMat(sx=1, sy=1, sz=1):
    scalMat = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz],
    ])
    return scalMat

def shiftMat(x, y, z):
    shiftMat = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return shiftMat

def keyPoints(face_landmark):
    up = face_landmark[:, 10]
    lp = face_landmark[:, 93]
    dp = face_landmark[:, 152]
    rp = face_landmark[:, 323]
    return up, lp, dp, rp


