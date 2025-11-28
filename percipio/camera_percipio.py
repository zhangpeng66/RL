"""
Provides the PercipioCamera class for capturing frames from Percipio cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from percipio.configuration_percipio import PercipioCameraConfig

import pcammls
from pcammls import * 
import sys
import os

logger = logging.getLogger(__name__)

class PythonPercipioDeviceEvent(pcammls.DeviceEvent):
    Offline = False

    def __init__(self):
        pcammls.DeviceEvent.__init__(self)

    def run(self, handle, eventID):
        if eventID==TY_EVENT_DEVICE_OFFLINE:
          print('=== Event Callback: Device Offline!')
          self.Offline = True
        return 0

    def IsOffline(self):
        return self.Offline

class PercipioCamera(Camera):

    def __init__(self, config: PercipioCameraConfig):
        """
        Initializes the PercipioCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        self.registration_mode = config.registration_mode
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.cl = None
        self.event = None
        self.handle = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)


    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.registration_mode})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.cl is not None and self.handle is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.cl = PercipioSDK()
        
        dev_list = self.cl.ListDevice()
 
        for idx in range(len(dev_list)):
            dev = dev_list[idx]
            print ('{} -- {} \t {}'.format(idx,dev.id,dev.iface.id))
        if  len(dev_list)==0:
            print ('no device')
            return
        if len(dev_list) == 1:
            selected_idx = 0 
        else:
            selected_idx  = int(input('select a device:'))
        if selected_idx < 0 or selected_idx >= len(dev_list):
            return

        sn = dev_list[selected_idx].id
        
        self.handle = self.cl.Open(sn)
        if not self.cl.isValidHandle(self.handle):
            err = self.cl.TYGetLastErrorCodedescription()
            print('no device found : ', end='')
            print(err)
            return

        self.event = PythonPercipioDeviceEvent()
        self.cl.DeviceRegiststerCallBackEvent(self.event)
        
        #该接口用于列举数据流的分辨率和图像格式。以 Color 数据流为例
        color_fmt_list = self.cl.DeviceStreamFormatDump(self.handle, PERCIPIO_STREAM_COLOR)
        if len(color_fmt_list) != 0:
            print ('color image format list:')
            for idx in range(len(color_fmt_list)):
                fmt = color_fmt_list[idx]
                print ('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, self.cl.Width(fmt), self.cl.Height(fmt), fmt.getDesc()))
                print('\tSelect {}'.format(fmt.getDesc()))
        #该接口用于配置数据流的分辨率，与 DeviceStreamFormatDump 联合使用
        self.cl.DeviceStreamFormatConfig(self.handle, PERCIPIO_STREAM_COLOR, color_fmt_list[0])


        #该接口用于设置相机的工作模式，0 代表 TY_TRIGGER_MODE_OFF，1 代表 TY_TRIGGER_MODE_SLAVE。示例如下：
        self.cl.DeviceControlTriggerModeEnable(self.handle, 1)
        #该接口用于加载相机的配置文件（custom_block.bin 文件中保存了相机参数）
        err = self.cl.DeviceLoadDefaultParameters(self.handle)
        if err:
            print('Load default parameters fail: ', end='')
            print(self.cl.TYGetLastErrorCodedescription())
        else:
            print('Load default parameters successful')
        #该接口用于使能数据流。使能 Color 和 Depth 数据流的示例如下：
        if self.use_depth:
            print('enable color and depth stream')
            err = self.cl.DeviceStreamEnable(self.handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
            if err:
                print('device stream enable err:{}'.format(err))
                return
        else:
            print('enable color stream')
            err = self.cl.DeviceStreamEnable(self.handle, PERCIPIO_STREAM_COLOR)
            if err:
                print('device stream enable err:{}'.format(err))
                return
        
        #开启数据流
        self.cl.DeviceStreamOn(self.handle)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras_info = []
        cl = PercipioSDK()
        
        dev_list = cl.ListDevice()
 
        if len(dev_list)==0:
            print ('no device')
            return found_cameras_info
        
        for idx in range(len(dev_list)):
            dev = dev_list[idx]
            print ('{} -- {} \t {}'.format(idx,dev.id,dev.iface.id))
            camera_info = {
                "id":dev.id,
                "dev.iface.id":dev.iface.id,
                "vendorName":dev.vendorName,
                "modelName":dev.modelName,
                "hardwareVersion":dev.hardwareVersion,
                "firmwareVersion":dev.firmwareVersion
            }
            found_cameras_info.append(camera_info)

        return found_cameras_info

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        if self.event.IsOffline():
            raise RuntimeError(f"{self}: device offline!")

        self.cl.DeviceControlTriggerModeSendTriggerSignal(self.handle)
        image_list = self.cl.DeviceStreamRead(self.handle, 20000)
        depth_render = image_data()
        depth_map_processed = []
        for i in range(len(image_list)):
            frame = image_list[i]
            arr = frame.as_nparray()
            if frame.streamID == PERCIPIO_STREAM_DEPTH:
               #该接口用于解析和渲染 Depth 图
               self.cl.DeviceStreamDepthRender(frame, depth_render)
               arr = depth_render.as_nparray()
               depth_map_processed = self._postprocess_image(arr, depth_frame=True)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        if self.event.IsOffline():
            raise RuntimeError(f"{self}: device offline!")

        self.cl.DeviceControlTriggerModeSendTriggerSignal(self.handle)
        image_list = self.cl.DeviceStreamRead(self.handle, 20000)
        rgb_image = image_data()
        color_image_processed = []
        for i in range(len(image_list)):
            frame = image_list[i]
            arr = frame.as_nparray()
            if frame.streamID == PERCIPIO_STREAM_COLOR:
               #该接口用于解析 Color 图
               self.cl.DeviceStreamImageDecode(frame, rgb_image)
               arr = rgb_image.as_nparray()
               color_image_processed = self._postprocess_image(arr, color_mode)


        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from RealSense).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    # NOTE(Steven): Missing implementation for depth for now
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the RealSense pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (pipeline not running).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

     
        self.cl.DeviceStreamOff(self.handle)    
        self.cl.Close(self.handle)

        logger.info(f"{self} disconnected.")
