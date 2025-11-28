from dataclasses import dataclass, field
from typing import Optional, Any
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import RobotConfig, Robot
from lerobot.utils.errors import DeviceNotConnectedError,DeviceAlreadyConnectedError

import __common

from jaka_zu5.config_jaka_zu5 import JAKAZU5Config

import numpy as np

import logging
import time
from functools import cached_property
from typing import Any

logger = logging.getLogger(__name__)

class JAKAZU5(Robot):
    config_class = JAKAZU5Config
    name = "jaka_zu5"

    def __init__(self, config: JAKAZU5Config):
        super().__init__(config)
        
        self.cameras = make_cameras_from_configs(config.cameras)

        # RTDE fields (hardware side)
        self.robot_ip = config.ip

        # servoJ streaming parameters
        # Use a finite period so the controller gracefully holds until next update
        # and does not require a perfect 125 Hz stream from Python/HTTP loop
        self.acc = 0.5
        self.speed = 0.5
        self.servoj_t = 1.0 / 500
        self.servoj_lookahead = 0.2
        self.servoj_gain = 100
        #initial jaka arm environment
        __common.init_env()
        import jkrc

        self.rc = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_0": float,
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.rc is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            self.rc = jkrc(self.robot_ip)
            self.rc.login()
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            return

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.rc:
            self.rc.logout()
            self.rc = None

        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        ret = self.rc.get_joint_position()
        joint_positions = []
        if ret[0] == 0:
            print("the joint position is :", ret[1])
            joint_positions = ret[1]
        else:
            print("some things happend,the errcode is: ",ret[0])
         
        obs_dict = {f"joint_{i}": val for i, val in enumerate(joint_positions)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # Check if action is valid
        if not all(key in self.action_features for key in action.keys()):
            raise ValueError(f"Invalid action: {action}, features: {self.action_features}")
        #机器人关节运动目标位置
        joint_pos = [action[f"joint_{i}"] for i in range(6)]
        #0 代表绝对运动，1 代表相对运动
        MODE = 0
        #设置接口是否为阻塞接口，TRUE 为阻塞接口 FALSE 为非阻塞接口，阻塞表示机器人运动完成才会有返回值，非阻塞表示接口调用完成立刻就有返回值。
        is_block = True
        #机器人关节运动速度，单位：rad/s
        SPEED = 0.2
        # Send goal position to the arm
        self.rc.joint_move(joint_pos, MODE, is_block, SPEED)

        return action