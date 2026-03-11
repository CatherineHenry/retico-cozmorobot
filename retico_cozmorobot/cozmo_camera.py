import os
# cozmo
import sys
import threading

# retico
import retico_core
from retico_core.robot import RobotStateIU
from retico_vision.vision import ImageIU

import cozmo
import time

from collections import deque


class CozmoCameraModule(retico_core.AbstractModule):
    '''
    use_viewer=True must be set in cozmo.run_program
    '''

    @staticmethod
    def name():
        return "Cozmo Camera Tracking Module"

    @staticmethod
    def description():
        return "A module that tracks cozmo camera frames."

    @staticmethod
    def input_ius():
        return [RobotStateIU]

    @staticmethod
    def output_iu():
        return ImageIU


    def __init__(self, robot, exposure=40, gain=2, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.robot.set_lift_height(0.0, in_parallel=True).wait_for_completed()
        self.exposure_amount = exposure
        self.gain_amount = gain
        self.configure_camera()

        self.handler = None
        self.img_queue = deque(maxlen=1)
        self.queue = deque()


    def process_update(self, update_message):

        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            while len(self.img_queue) < 1:
                time.sleep(0.05)
            img = self.img_queue.popleft()
            output_iu = self.create_iu(input_iu)
            output_iu.set_image(img, 1, 1)
            self.robot.camera.image_stream_enabled = False
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)
        return None

    def configure_camera(self):
        self.robot.camera.image_stream_enabled = True
        self.robot.camera.color_image_enabled = True
        self.robot.camera.enable_auto_exposure(True) # = False # False means we can adjust manually
        time.sleep(5) # wait for these settings to propagate through to Cozmo
        self.robot.camera.enable_auto_exposure(False) # = False # False means we can adjust manually

        # Lerp exposure between min and max times
        min_exposure = self.robot.camera.config.min_exposure_time_ms
        max_exposure = self.robot.camera.config.max_exposure_time_ms
        trimmed_exposure = max(min_exposure, min(self.exposure_amount, max_exposure))
        # Lerp gain
        min_gain = self.robot.camera.config.min_gain
        max_gain = self.robot.camera.config.max_gain
        trimmed_gain =  max(min_gain, min(self.gain_amount, max_gain))
        print(f"[Before Setting] Exposure: {self.robot.camera.exposure_ms}, Gain: {self.robot.camera.gain}")

        self.robot.camera.set_manual_exposure(trimmed_exposure,trimmed_gain)
        time.sleep(5) # wait for these settings to propagate through to Cozmo
        # Setting twice, I've found that sometimes the first set doesn't reliably apply
        self.robot.camera.set_manual_exposure(trimmed_exposure,trimmed_gain)
        print(f"[After Setting] Exposure: {self.robot.camera.exposure_ms}, Gain: {self.robot.camera.gain}")


    def prepare_run(self):
        def handle_image(evt, obj=None, tap_count=None,  **kwargs):
            self.img_queue.append(evt.image)

        self.handler = self.robot.world.add_event_handler(cozmo.camera.EvtNewRawCameraImage, handle_image)

        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        if self.handler is not None:
            self.handler.disable()
        self._extractor_thread_active = False

