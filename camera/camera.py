import numpy as np
import math


class Camera:
    def __init__(self, frame_width, frame_height, f, size_px, x, y, z, x_rotation_angle, y_rotation_angle):
        self.intrinsic_matrix = create_intrinsic_matrix(frame_width, frame_height, f, size_px)
        self.rotation_matrix = create_rotation_matrix(x_rotation_angle, y_rotation_angle)
        self.translation_vector = create_translation_vector(x, y, z)
        self.extrinsic_matrix = create_extrinsic_matrix(
            self.rotation_matrix,
            np.dot(self.rotation_matrix, self.translation_vector)
        )
        self.p_matrix = create_p_matrix(self.intrinsic_matrix, self.extrinsic_matrix)


def create_intrinsic_matrix(width, height, f, size_px):
    return np.array([[f / size_px, 0, width / 2],
                     [0, f / size_px, height / 2],
                     [0, 0, 1]])


def create_rotation_matrix(x_rotation_angle, y_rotation_angle):
    R = np.array([[0, 0, 1],
                  [0, -1, 0],
                  [1, 0, 0]])
    return np.dot(np.dot(get_rotation_z_matrix(0),
                         np.dot(get_rotation_x_matrix(-x_rotation_angle),
                                get_rotation_y_matrix(-y_rotation_angle))), R)


def create_p_matrix(intrinsic_matrix, extrinsic_matrix):
    return np.dot(intrinsic_matrix, extrinsic_matrix)


def create_translation_vector(x, y, z):
    return np.array([[x], [y], [z]])


def create_extrinsic_matrix(rotation_matrix, translation_vector):
    return np.concatenate((rotation_matrix, translation_vector), axis=1)


def create_extrinsic_matrix_for_main_camera():
    rotation_matrix = np.eye(3)
    translation_vector = np.array([[0, 0, 0]])
    return create_extrinsic_matrix(rotation_matrix, translation_vector)


def get_rotation_x_matrix(angle):
    angle = angle * math.pi / 180
    return np.array([[1, 0, 0],
                     [0, math.cos(angle), -math.sin(angle)],
                     [0, math.sin(angle), math.cos(angle)]])


def get_rotation_y_matrix(angle):
    angle = angle * math.pi / 180
    return np.array([[math.cos(angle), 0, math.sin(angle)],
                     [0, 1, 0],
                     [-math.sin(angle), 0, math.cos(angle)]])


def get_rotation_z_matrix(angle):
    angle = angle * math.pi / 180
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])
