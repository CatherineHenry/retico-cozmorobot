import functools
import threading
import time
import asyncio
import sys

import cv2

# retico
import retico_core
from retico_core.robot import IACMotorAction
from retico_vision.vision import ImageIU

# cozmo
import sys
import os
sys.path.append(os.environ['COZMO'])
import cozmo
# import cv2
import time

from collections import deque
import numpy as np
from PIL import Image
import base64



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
        return [IACMotorAction]

    @staticmethod
    def output_iu():
        return ImageIU


    def __init__(self, robot, exposure=0.05, gain=0.1, **kwargs):
        # for exp room:exposure=0.05, gain=0.05
        super().__init__(**kwargs)
        self.robot = robot
        # My cozmo isn't reliably able to lift the lift up so set down to 0 for now. - Catherine (its not obstructing the camera so its fine)
        self.robot.set_lift_height(0.0, in_parallel=True).wait_for_completed() # this doesnt work for lifting  up for some reason
        # self.robot.move_lift(5) # this locks the track cannot do other stuff later
        # use this to lift the lift because it can be stopped later. set lift height doesnt work for some reason.
        # self.robot.play_anim_trigger(cozmo.anim.Triggers.CubePounceIdleLiftUp).wait_for_completed()        self.exposure_amount = exposure
        self.exposure_amount = exposure
        self.gain_amount = gain
        self.img_queue = deque(maxlen=1)
        self.queue = deque()

        # NOTE: was seeing intermittent issues when this was in setup -- the exposure/gain was not setting correctly and would be too bright
        # self.configure_camera()

    def process_update(self, update_message):

        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

        # if len(self.img_queue) > 0:
        #     img = self.img_queue.popleft()
        #     # img = np.array(img)
        #     output_iu = self.create_iu(None)
        #     output_iu.set_image(img, 1, 1)
        #     return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)


            # bytes = img.tobytes()
            # output_iu.set_image(base64.b64encode(bytes).decode(), 1, 1)
            # return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            try:
                img = self.img_queue.popleft()
            except IndexError:
                time.sleep(1)
                img = self.img_queue.popleft()
            # img = np.array(img)
            output_iu = self.create_iu(input_iu)
            # img = np.array(img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img)
            output_iu.set_image(img, 1, 1)
            output_iu.set_flow_uuid(input_iu.flow_uuid)
            output_iu.set_execution_uuid(input_iu.execution_uuid)
            output_iu.set_motor_action(input_iu.motor_action)
            self.robot.camera.image_stream_enabled = False
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

        #     return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        return None

    def configure_camera(self):
        self.robot.camera.color_image_enabled = True
        self.robot.camera.enable_auto_exposure = False # False means we can adjust manually
        # Lerp exposure between min and max times
        min_exposure = self.robot.camera.config.min_exposure_time_ms
        max_exposure = self.robot.camera.config.max_exposure_time_ms
        exposure_time = (1 - self.exposure_amount) * min_exposure + self.exposure_amount * max_exposure
        # Lerp gain
        min_gain = self.robot.camera.config.min_gain
        max_gain = self.robot.camera.config.max_gain
        actual_gain = (1-self.gain_amount)*min_gain + self.gain_amount*max_gain
        self.robot.camera.set_manual_exposure(exposure_time,actual_gain)

    def prepare_run(self):
        def handle_image(evt, obj=None, tap_count=None,  **kwargs):
            self.img_queue.append(evt.image)

        self.configure_camera()
        time.sleep(20)
        self.robot.world.add_event_handler(cozmo.camera.EvtNewRawCameraImage, handle_image)

        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False

