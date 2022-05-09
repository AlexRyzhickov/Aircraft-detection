import numpy as np
import math


class Camera:
    def __init__(self, extrinsic_matrix, intrinsic_matrix):
        self.extrinsic_matrix = extrinsic_matrix
        self.intrinsic_matrix = intrinsic_matrix


def create_intrinsic_matrix(width, height, f, size_px):
    return np.array([[f / size_px, 0, width / 2],
                     [0, f / size_px, height / 2],
                     [0, 0, 1]])


def create_rotation_matrix(angle, beta):
    alpha = angle * math.pi / 180
    beta = beta * math.pi / 180
    return np.array([[-math.sin(math.pi / 18), 0, math.cos(math.pi / 18)],
                     [math.sin(math.pi / 36) * math.cos(math.pi / 18), - math.cos(math.pi / 36),
                      math.sin(math.pi / 36) * math.sin(math.pi / 18)],
                     [math.cos(math.pi / 36) * math.cos(math.pi / 18), math.sin(math.pi / 36),
                      math.cos(math.pi / 36) * math.sin(math.pi / 18)]])


def create_translation_vector(x, y, z):
    return np.array([[x], [y], [z]])


def create_extrinsic_matrix(rotation_matrix, translation_vector):
    return np.concatenate((rotation_matrix, translation_vector.T), axis=1)


def create_extrinsic_matrix_for_main_camera():
    rotation_matrix = np.eye(3)
    translation_vector = np.array([[0, 0, 0]])
    return create_extrinsic_matrix(rotation_matrix,translation_vector)
