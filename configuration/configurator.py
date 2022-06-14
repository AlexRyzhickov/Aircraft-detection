import json
import camera.camera as cm
from enum import Enum

class BgMethod(Enum):
    KNN = 1
    CNT = 2
    MOG = 3
    MOG2 = 4


class CameraConfigurations:
    def __init__(self,
                 name: str,
                 path: str,
                 frame_width: int,
                 frame_height: int,
                 f: float,
                 size_px: float,
                 x: float,
                 y: float,
                 z: float,
                 x_rotation_angle: float,
                 y_rotation_angle: float,
                 z_rotation_angle: float
                 ):
        self.name = name
        self.path = path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = f
        self.size_px = size_px
        self.x = x
        self.y = y
        self.z = z
        self.x_rotation_angle = x_rotation_angle
        self.y_rotation_angle = y_rotation_angle
        self.z_rotation_angle = z_rotation_angle


class SceneConfigurations:
    def __init__(self,
                 y_plane: float,
                 v_offset: float,
                 h_offset: float,
                 buff_size: int,
                 frames_per_second: int,
                 contour_min_size: int,
                 contour_large_size: int,
                 size_difference: float,
                 close_range: int,
                 bg_method: BgMethod
                 ):
        self.y_plane = y_plane
        self.v_offset = v_offset
        self.h_offset = h_offset
        self.buff_size = buff_size
        self.frames_per_second = frames_per_second
        self.contour_min_size = contour_min_size
        self.contour_large_size = contour_large_size
        self.size_difference = size_difference
        self.close_range = close_range
        self.bg_method = bg_method


def get_cameras_configurations(filepath: str):
    with open(filepath) as json_data:
        data = json.load(json_data)
        cameras = data["cameras"]

        if len(cameras) < 2:
            raise Exception("The minimum number of cameras is less than two")

        cameras_configurations = []

        for camera in cameras:
            camera_conf = CameraConfigurations(
                camera["name"],
                camera["video_path"],
                camera["frame_width"],
                camera["frame_height"],
                camera["focus"],
                camera["size_px"],
                camera["x"],
                camera["y"],
                camera["z"],
                camera["x_rotation_angle"],
                camera["y_rotation_angle"],
                camera["z_rotation_angle"],
            )
            cameras_configurations.append(camera_conf)

        paths = [camera_conf.path for camera_conf in cameras_configurations]
        names = [camera_conf.name for camera_conf in cameras_configurations]
        cameras = [cm.Camera(camera_conf.frame_width,
                             camera_conf.frame_height,
                             camera_conf.focus,
                             camera_conf.size_px,
                             camera_conf.x,
                             camera_conf.y,
                             camera_conf.z,
                             camera_conf.x_rotation_angle,
                             camera_conf.y_rotation_angle
                             )
                   for camera_conf in cameras_configurations]
        return paths, names, cameras


def analyze_bg_method_str(method_name: str):
    if method_name == "KNN":
        return BgMethod.KNN
    elif method_name == "CNT":
        return BgMethod.CNT
    elif method_name == "MOG":
        return BgMethod.MOG
    elif method_name == "MOG2":
        return BgMethod.MOG2
    elif method_name == None:
        return BgMethod.CNT
    raise Exception("Incorrect bg method name")


def get_scene_configurations(filepath: str):
    with open(filepath) as json_data:
        data = json.load(json_data)

        scene_conf = SceneConfigurations(
            data["y_plane"],
            data["v_offset"],
            data["h_offset"],
            data["buff_size"],
            data["frames_per_second"],
            data["contour_min_size"],
            data["contour_large_size"],
            data["size_difference"],
            data["close_range"],
            analyze_bg_method_str(data.get("bg_method"))
        )

        return scene_conf
