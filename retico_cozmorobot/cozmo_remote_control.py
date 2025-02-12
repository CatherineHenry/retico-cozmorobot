import threading
import time
from collections import deque

import numpy as np
from flask import Flask

# retico
import retico_core
from retico_core.robot import IACMotorAction
from retico_cozmorobot.cozmo_remote_control_flask_app import RemoteControlCozmo

flask_app = Flask(__name__)


class CozmoRemoteControlModule(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "Cozmo Remote Control Module"

    @staticmethod
    def description():
        return "A module that allows for manually driving cozmo, outputting a pose on completion."

    @staticmethod
    def input_ius():
        return [IACMotorAction]

    @staticmethod
    def output_iu():
        return IACMotorAction  # Instead of coming from IAC module, comes from here now (when manually controlled)


    def __init__(self, robot, exposure=0.05, gain=0.1, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.img_queue = deque(maxlen=1)
        self.queue = deque()
        # Start Flask server in background
        self.remote_control_cozmo = RemoteControlCozmo(self.robot)
        threading.Thread(target=self.remote_control_cozmo.run, args=[self.remote_control_cozmo]).start()

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
            output_iu = self.create_iu(input_iu)

            # Re-enable image stream, this might cause issues with popping an image from queue that was from time moving
            self.robot.camera.image_stream_enabled = True
            # buttons are disabled on click, need to re-enable when back here ready for a pose
            self.remote_control_cozmo.enable_savepose_button()
            while len(self.remote_control_cozmo.pose_queue) == 0:
                time.sleep(0.1)
                continue
            robot_pose = self.remote_control_cozmo.pose_queue.popleft()
            # x,y (width and length of space + rotation) (ndarray to align with iu used in other executions)
            motor_action = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.rotation.angle_z.degrees])
            output_iu.set_motor_action(motor_action=motor_action, flow_uuid=input_iu.flow_uuid, execution_uuid=input_iu.execution_uuid)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)
        return None

    def prepare_run(self):
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False

