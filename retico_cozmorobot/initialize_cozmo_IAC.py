import logging
import time
import tkinter as tk
import uuid

import cozmo
import numpy as np
from explauto import InterestModel, SensorimotorModel
from explauto.agent import ReticoAgent

import retico_core
from retico_core import abstract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IACInitializationIU(abstract.IncrementalUnit):
    @staticmethod
    def type():
        return "IAC initialization IU"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in)
        self.payload = None

    def set_metadata(self, execution_uuid, save_data, experiment_name, manual_control, flow_uuid, date_timestamp=None):
        self.meta_data = {
            'execution_uuid': execution_uuid,
            'manual_control': manual_control,
            'save_data': save_data,
            'experiment_name': experiment_name,
            'flow_uuid': flow_uuid,
            'date_timestamp': date_timestamp # should only be set if the execution_uuid is set
        }

    def set_payload(self, motor_goal: []):
        """
        Sets the motor goal taken so we can move the robot to the specified goal on the client side
        """
        self.payload = motor_goal


class CozmoIntelligentAdaptiveCuriosityInitializationModule(abstract.AbstractProducingModule, tk.Frame):
    """
    use_viewer=True must be set in cozmo.run_program
    A short video of this module in action can be found on YouTube https://youtu.be/iUKjYkx-IFY
    """

    @staticmethod
    def name():
        return "Cozmo Intelligent Adaptive Curiosity"

    @staticmethod
    def description():
        return "A module that runs initialized Explauto + Cozmo, passing importing information for all future IUs via the output IU metadata"

    @staticmethod
    def output_iu():
        return IACInitializationIU

    def __init__(self, robot: cozmo.robot.Robot, experiment_name, save_data=False, execution_uuid=None, manual_control=True, date_timestamp=None, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot

        # Metadata passed to all future IUs, these values should be static
        self.save_data = save_data
        self.execution_uuid = execution_uuid
        self.manual_control = manual_control
        self.experiment_name = experiment_name

        # Passing through if the execution_uuid is set, so we can log what day it was originally from
        # and for copying prior execution data
        self.date_timestamp = date_timestamp


    def process_update(self, update_message):
        output_iu = self.create_iu(grounded_in=None)

        init_robot_position = np.array([self.robot.pose.position.x, self.robot.pose.position.y, self.robot.pose.rotation.angle_z.degrees])
        flow_uuid = 'init_' + str(uuid.uuid4()).split("-")[0]
        output_iu.set_payload(motor_goal=init_robot_position)
        output_iu.set_metadata(execution_uuid=self.execution_uuid, save_data=self.save_data, experiment_name=self.experiment_name,
                               manual_control=self.manual_control, flow_uuid=flow_uuid,
                               date_timestamp=self.date_timestamp)


        # Break out of producer loop after running 1x, we only need this for basic initialization
        self.stop()
        print("Stopped loop for Cozmo IAC Init module")
        # The Producer Modules use return instead of um.append? or did I mess something up
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
