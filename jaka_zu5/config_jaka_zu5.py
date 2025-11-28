from dataclasses import dataclass, field

from lerobot.cameras.configs  import CameraConfig

from lerobot.robot.config import RobotConfig


@RobotConfig.register_subclass("jaka_zu5")
@dataclass
class JAKAZU5Config(RobotConfig):
    #IP to connect to the arm
    ip: str
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)