import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from explauto import InterestModel, SensorimotorModel
from explauto.agent import ReticoAgent

from explauto.environment.cozmo_env import CozmoEnvironment
import tkinter as tk



from cozmo.util import Pose, degrees, Angle, distance_mm, speed_mmps

import retico_core
from retico_core import abstract, UpdateType
from retico_cozmorobot.cozmo_state import RobotStateIU
from retico_core.robot import IACMotorAction
from retico_vision.vision import ObjectFeaturesIU

sys.path.append(os.environ["COZMO"])
import cozmo

from collections import deque
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentName(Enum):
    # Note: These values need to stay in sync with the Explauto interest model config names
    a = 'cozmo_binary_obj_detection'  # implementation with minimal changes to compare against prior work. Only include T/F obj detected. No other changes.
    b = 'cozmo_clip'  # include clip feature vector but don't make any other changes
    c = 'cozmo_clip_cos_sim_split'  # include clip and split region by cos similarity


    d = 'cozmo_clip_cos_split_and_learning_progress'  # include clip, split region by cos similarity, and adjust learning progress calculation


class CozmoIntelligentAdaptiveCuriosityModule(abstract.AbstractModule, tk.Frame):
    """
    use_viewer=True must be set in cozmo.run_program
    A short video of this module in action can be found on YouTube https://youtu.be/iUKjYkx-IFY
    """

    @staticmethod
    def name():
        return "Cozmo Intelligent Adaptive Curiosity"

    @staticmethod
    def description():
        return "A module that runs Explauto + Cozmo"

    @staticmethod
    def input_ius():
        return [RobotStateIU, ObjectFeaturesIU]

    @staticmethod
    def output_iu():
        return IACMotorAction

    def __init__(self, robot: cozmo.robot.Robot, tk_root, date_timestamp, experiment_name, agent=None, save_data=False, execution_uuid=None, max_turn_count=0, **kwargs):
        super().__init__(**kwargs)
        self.num_ius_processed = 0
        self.robot = robot
        self.init_pose = robot.pose
        self.center_pose = Pose(350, 180, 0, angle_z=Angle(0))
        self.queue = deque(maxlen=1)
        self.motors = None  # TBD how to do this. Probably make our own objects for the misc motors (tread, head, etc)
        self.move_duration = 2.0  # The move duration (in s)
        self.tracked_obj = None  # TBD. Seems to be only for use with vrep? expects string.
        self.date_timestamp = date_timestamp
        self.save_data = save_data
        self.experiment_shorthand_name = ExperimentName(experiment_name).name
        self.experiment_name = experiment_name
        self.configured_gain = 0.2942187514156103
        self.configured_exposure_ms = 34.0
        self.execution_uuid = execution_uuid
        self.max_turn_count = max_turn_count

        if experiment_name == ExperimentName.a.value:  # without CLIP all we have is a T/F binary flag for if an obj is detected or not
            self.sensory_space_size = 1  # T or False binary value
        else:  # including CLIP
            self.sensory_space_size = 384  # TODO: add location info will consist of obj relative location + size feats concat w clip output

        if agent is None:
            print(f"Starting new execution with uuid {self.execution_uuid} and date {self.date_timestamp}")
            m_mins = [-180, -80]  # angle of rotation, backward linear travel
            m_maxs = [180, 80]  # angle of rotation, forward linear travel
            # m_mins = [-130, -80]  # angle of rotation, backward linear travel. Zone out cube location
            # m_maxs = [130, 80]  # angle of rotation, forward linear travel. Zone out cube location



            s_mins = [-1] * self.sensory_space_size  # -1 because 0 is a valid CLIP output
            s_maxs = [1] * self.sensory_space_size

            self.cozmo_env = CozmoEnvironment(
                cozmo_robot=robot,
                move_duration=self.move_duration,
                tracker=None,  # want this to be the result of the iu?
                m_mins=m_mins,
                m_maxs=m_maxs,
                s_mins=s_mins,
                s_maxs=s_maxs,
            )
            self.sensorimotor_model = SensorimotorModel.from_configuration(self.cozmo_env.conf, 'LWLR-NONE', 'default')
            # self.sensorimotor_model = SensorimotorModel.from_configuration(self.cozmo_env.conf, 'NSLWLR-NONE', 'default')
            # Select Interest Model config based on Experiment
            config_name = experiment_name
            self.interest_model = InterestModel.from_configuration(self.cozmo_env.conf, self.cozmo_env.conf.m_dims, 'tree', config_name)
            self.agent = ReticoAgent(self.cozmo_env.conf, self.sensorimotor_model, self.interest_model, execution_uuid=self.execution_uuid, execution_date_timestamp=self.date_timestamp, save_data=self.save_data, experiment_name=experiment_name)  # agent is necessary to avoid bootstrapping issues

        else:
            self.agent = agent
            print(f"Loading prior execution with uuid {self.execution_uuid} and date {self.date_timestamp}. Continuing with experiment '{experiment_name}'")
            self.cozmo_env = CozmoEnvironment(
                cozmo_robot=robot,
                move_duration=self.move_duration,
                tracker=None,  # want this to be the result of the iu?
                m_mins=self.agent.conf.m_mins,
                m_maxs=self.agent.conf.m_maxs,
                s_mins=self.agent.conf.s_mins,
                s_maxs=self.agent.conf.s_maxs,
            )
            self.interest_model = self.agent.interest_model
            self.sensorimotor_model = self.agent.sensorimotor_model
        self.time_slept = 0


    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _extractor_thread(self):
        # take in feature, update learning. output new action, this should kick off camera which will result in new feature input
        # first feature should just be whatever action 0 was, or maybe just throw away first and do action
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            if len(self.queue) == 0:
                time.sleep(0.5)
                self.time_slept += 0.5
                # continue

                if self.time_slept >= 200:
                    input_iu = ObjectFeaturesIU()
                    input_iu.set_object_features(image=None, object_features={'0': [-1] * self.sensory_space_size})
                    self.time_slept = 0

                else:
                    continue

            else:
                input_iu = self.queue.popleft()
                self.time_slept = 0
            # when new objects are observed (i.e., not SpeechRecognitionIUs)
            if isinstance(input_iu, ObjectFeaturesIU):
                output_iu = self.create_iu(grounded_in=input_iu)
                motor_action = input_iu.motor_action
                objects = input_iu.payload
                if len(objects) == 0:
                    print("Didn't get feature, setting to -1 and continuing.")
                    sensori_effect = [-1]*self.sensory_space_size
                else:
                    if self.sensory_space_size == 1: # ignore the CLIP output and flag as "1" for obj detected
                        sensori_effect = [1]
                    else:
                        sensori_effect = input_iu.payload["0"]

                if '0' in input_iu.grounded_in.payload:
                    label = 'something'
                else:
                    label = 'whitespace'

                inferred_sensori = self.agent.y

                if self.save_data:
                    sensori_df = pd.DataFrame.from_records([np.hstack([motor_action, sensori_effect]), np.hstack([motor_action, inferred_sensori])])
                    sensori_df.insert(0, 'expl_dims', [len(self.agent.expl_dims)]*2)
                    sensori_df.insert(0, 'inf_dims', [len(self.agent.inf_dims)]*2)
                    sensori_df.insert(0, 'experiment_name', self.experiment_name)
                    sensori_df.insert(0, 'sensori_type', ['effect', 'inferred'])
                    sensori_df.insert(0, 'obj_name', [label]*2)
                    sensori_df.insert(0, 'flow_uuid', [input_iu.flow_uuid]*2)
                    sensori_df.insert(0, 'exec_uuid', [self.execution_uuid]*2)


                    sensori_df.to_csv(f'./IAC_output_data/sensori_effect_{self.date_timestamp}_{self.execution_uuid}_{self.experiment_shorthand_name}.csv', mode='a', index=False, header=False)


                
                # inform the agent of the sensorimotor consequence of the action and update both the sensorimotor and interest models
                self.agent.perceive(sensori_effect, flow_uuid=input_iu.flow_uuid)



                undo_drive = -motor_action[1]
                self.robot.drive_straight(distance_mm(undo_drive), speed_mmps(105), should_play_anim=False).wait_for_completed()
                undo_rotation = -motor_action[0]
                self.robot.turn_in_place(degrees(undo_rotation), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed() # slow so it's more accurate

                if self.num_ius_processed > 0 and self.num_ius_processed % 5 == 0:  # Every 5 actions
                    self.robot.set_robot_volume(0.1) # 1 to hear at home
                    self.robot.say_text("Re centering", duration_scalar=0.95, voice_pitch=1, use_cozmo_voice=True).wait_for_completed()
                    self.robot.camera.color_image_enabled = False  # TODO: adjust gain and exposure, if not finding lightcube symbol reliably
                    # self.robot.camera.set_manual_exposure(34.0, 0.21653125144541263)

                    self.robot.set_robot_volume(0)
                    self.robot.turn_in_place(degrees(180), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed()
                    # self.robot.turn_in_place(degrees(-90), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed()
                    self.robot.drive_straight(distance_mm(150), speed_mmps(105), should_play_anim=False).wait_for_completed()
                    action = None
                    recenter_attempts = 0
                    while action is None or action.state !='action_succeeded' and recenter_attempts <=3:
                        look_around = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)

                        try:
                            cube = self.robot.world.wait_for_observed_light_cube(timeout=60, include_existing=False)
                        except asyncio.TimeoutError:
                            print(f"Didn't find a cube :-( on try {recenter_attempts}")
                        finally:
                            look_around.stop()

                        # Cozmo will approach the cube he has seen
                        # using a 180 approach angle will cause him to drive past the cube and approach from the opposite side
                        # num_retries allows us to specify how many times Cozmo will retry the action in the event of it failing
                        action = self.robot.dock_with_cube(cube, approach_angle=cozmo.util.degrees(180), num_retries=2).wait_for_completed()

                        if action.state != 'action_succeeded':
                            recenter_attempts += 1
                            self.robot.drive_straight(distance_mm(-50), speed_mmps(60), should_play_anim=False).wait_for_completed()
                            # self.robot.drive_straight(distance_mm(-80), speed_mmps(60), should_play_anim=False).wait_for_completed()
                    # self.robot.camera.set_manual_exposure(self.configured_exposure_ms, self.configured_gain)
                    self.robot.camera.color_image_enabled = True

                    self.robot.drive_straight(distance_mm(-275), speed_mmps(105), should_play_anim=False).wait_for_completed()
                    # self.robot.drive_straight(distance_mm(-145), speed_mmps(105), should_play_anim=False).wait_for_completed()
                    self.robot.turn_in_place(degrees(180), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed()
                    # self.robot.turn_in_place(degrees(90), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed()
                    time.sleep(2)
                    # self.robot.play_anim_trigger(cozmo.anim.Triggers.CubePounceIdleLiftUp).wait_for_completed()
                    self.robot.set_head_angle(degrees(10), accel=10.0, max_speed=10.0, duration=1,
                                         warn_on_clamp=True, in_parallel=True, num_retries=2).wait_for_completed()
                    self.robot.set_lift_height(0).wait_for_completed()


                self.robot.camera.image_stream_enabled = True
                time.sleep(0.2)  # will too short a delay result in no image to pop in camera IU?

                # if self.robot.battery_voltage < 3.5: # per docs, 3.5 is low. Not linear.
                #     self.robot.camera.image_stream_enabled = False
                #     self.robot.set_robot_volume(5)
                #     self.robot.say_text("Charge me", duration_scalar=0.95, voice_pitch=1, use_cozmo_voice=True).wait_for_completed()
                #     time.sleep(300)  # 5 minutes
                #     self.robot.say_text("Ready", duration_scalar=0.95, voice_pitch=1, use_cozmo_voice=True).wait_for_completed()
                #     self.robot.set_robot_volume(0)
                #     time.sleep(60)

                turn_count = len(self.interest_model.data_x)
                # We've completed max number of turns, save the model and exit
                if self.max_turn_count != 0 and turn_count == self.max_turn_count:
                    self.agent.save(f"./IAC_output_data/agent_{self.execution_uuid}.pickle")
                    sys.exit()

                flow_uuid4 = str(uuid.uuid4()).split("-")[0]
                motor_goal = self.agent.produce(flow_uuid=flow_uuid4)

                self.cozmo_env.update(motor_goal, log=False)

                output_iu.set_motor_action(motor_action=motor_goal, flow_uuid=flow_uuid4, execution_uuid=self.execution_uuid)
                um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                self.append(um)
                self.num_ius_processed += 1



    def prepare_run(self):
        # `produce` calls `sample()` method on interest model then uses the sensorimotor model to obtain sensorimotor
        # vector using forward prediction (if motor babbling). Returns the motor part of the full sensorimotor vector.
        flow_uuid4 = str(uuid.uuid4()).split("-")[0]
        motor_goal = self.agent.produce(flow_uuid=flow_uuid4)
        # Execute the motor goal. We cannot get the sensori effect yet.
        self.cozmo_env.update(motor_goal, log=False)

        output_iu = self.create_iu(grounded_in=None)
        output_iu.set_motor_action(motor_action=motor_goal, flow_uuid=flow_uuid4, execution_uuid=self.execution_uuid)
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)
        # t = threading.Thread(target=self._process_iu_helper, name="cozmo_obj_permanence_process_iu_helper")
        # t.start()
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False
