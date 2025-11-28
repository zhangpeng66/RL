from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation

@CameraConfig.register_subclass("Percipio")
@dataclass
class PercipioCameraConfig(CameraConfig):
    """Configuration class for Percipio-based camera devices.

    Attributes:
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)

    Note:

    """

    use_depth: bool = False
    registration_mode: bool = False
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        values = (self.use_depth, self.registration_mode)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `use_depth`, `registration_mode`  either all of them need to be set, or none of them."
            )