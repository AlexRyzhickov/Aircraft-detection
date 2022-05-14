import numpy as np
import camera.camera as cm


def create_fundamental_matrix(camera1: cm.Camera, camera2: cm.Camera):
    T = create_t_vector(camera1.translation_vector, camera2.translation_vector)
    T_cross = create_t_cross_matrix(T)
    E = np.dot(camera1.rotation_matrix, np.dot(T_cross, np.transpose(camera2.rotation_matrix)))
    F = np.dot(np.linalg.inv(np.transpose(camera1.intrinsic_matrix)),
               np.dot(E, np.linalg.inv(camera2.intrinsic_matrix)))
    return F


def create_t_cross_matrix(T):
    return np.array([[0, - T[2], T[1]],
                     [T[2], 0, - T[0]],
                     [-T[1], T[0], 0]])


def create_t_vector(translation_vector1, translation_vector2):
    return (translation_vector2 - translation_vector1).reshape(3)
