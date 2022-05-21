import json
import camera.camera as cm


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


def get_cameras_configurations(filepath: str):
    with open(filepath) as json_data:
        data = json.load(json_data)
        cameras = data["cameras"]

        if len(cameras) < 2:
            raise Exception('The minimum number of cameras is less than two')

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
