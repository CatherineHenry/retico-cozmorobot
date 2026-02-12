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
        self.flow_uuid = None

    def set_metadata(self, execution_uuid, save_data, experiment_name, max_turn_count, manual_control):
        self.meta_data = {
            'execution_uuid': execution_uuid,
            'manual_control': manual_control,
            'save_data': save_data,
            'max_turn_count': max_turn_count,
            'experiment_name': experiment_name,
            # 'date_timestamp': date_timestamp,
            # 'experiment_shorthand_name': experiment_shorthand_name,
        }

    def set_payload(self, motor_goal: []):
        """
        Sets the motor goal taken so we can move the robot to the specified goal on the client side
        """
        self.payload = motor_goal
        # self.agent = agent
        self.flow_uuid = 'init_' + str(uuid.uuid4()).split("-")[0] # TODO: can I update these in the meta data? will need to be able to update for each cycle


#
# class ExperimentName(Enum):
#     # Note: These values need to stay in sync with the Explauto interest model config names
#     a = 'cozmo_binary_obj_detection'  # implementation with minimal changes to compare against prior work. Only include T/F obj detected. No other changes.
#     b = 'cozmo_clip'  # include clip feature vector but don't make any other changes
#     c = 'cozmo_clip_cos_sim_split'  # include clip and split region by cos similarity
#
#
#     d = 'cozmo_clip_cos_split_and_learning_progress'  # include clip, split region by cos similarity, and adjust learning progress calculation
#
#     e = 'cozmo_clip_cos_sim_split_random_sampling'
#     f = 'cozmo_clip_cos_sim_split_with_region_deletion'

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

    def __init__(self, robot: cozmo.robot.Robot, experiment_name, save_data=False, execution_uuid=None, max_turn_count=0, manual_control=True, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot

        # Metadata passed to all future IUs
        self.save_data = save_data
        self.execution_uuid = execution_uuid
        self.max_turn_count = max_turn_count
        self.manual_control = manual_control
        self.experiment_name = experiment_name
        # self.experiment_shorthand_name = ExperimentName(self.experiment_name).name
        # self.date_timestamp = date_timestamp # is set in runner (either loaded from agent or new) so saves on client side go to correct directory

        # All of this data *could* come from a loaded agent from a prior save. Try doing server side
        # self.experiment_shorthand_name = ExperimentName(experiment_name).name
        # self.rand_seed = rand_seed if rand_seed is not None else np.random.randint(100000)
        # print(f"Random seed is {self.rand_seed}")

        # sent over ZMQ (so we can load prior agents
        # self.agent = agent # TODO: can we pickle the loaded agent?



    def process_update(self, update_message):
        output_iu = self.create_iu(grounded_in=None)

        init_robot_position = np.array([self.robot.pose.position.x, self.robot.pose.position.y, self.robot.pose.rotation.angle_z.degrees])

        # TODO: set these as meta data instead. As well as save data, random seed, etc. Everything we need on server side IAC
        output_iu.set_payload(motor_goal=init_robot_position)
        output_iu.set_metadata(execution_uuid=self.execution_uuid, save_data=self.save_data, experiment_name=self.experiment_name,
                               max_turn_count=self.max_turn_count, manual_control=self.manual_control,)


        self.stop()
        print("Stopped loop for Cozmo IAC Init module")
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        # The Producer Modules use return instead of queue..?
        # Try this without the stop and se
        # um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        # self.append(um)
        # TODO: move stop to above return
        time.sleep(5) # Give time for IU to make it to the camera module then stop
        # Break out of producer loop after running 1x, we only need this for basic initialization
