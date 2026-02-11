import asyncio
import copy
import os
import pickle
import sys
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw
from explauto import InterestModel, SensorimotorModel
from explauto.agent import ReticoAgent

from explauto.environment.cozmo_env import CozmoEnvironment
import tkinter as tk



from cozmo.util import Pose, degrees, Angle, distance_mm, speed_mmps

import retico_core
from retico_core import abstract, UpdateType
from retico_core.robot import IACMotorAction
from retico_vision.vision import ObjectFeaturesIU, ObjectPermanenceIU

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

    e = 'cozmo_clip_cos_sim_split_random_sampling'
    f = 'cozmo_clip_cos_sim_split_with_region_deletion'

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
        return [ObjectPermanenceIU]

    @staticmethod
    def output_iu():
        return IACMotorAction

    def __init__(self, robot: cozmo.robot.Robot, date_timestamp, experiment_name, agent=None, save_data=False, execution_uuid=None, max_turn_count=0, manual_control=True, rand_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.num_ius_processed = 0
        self.robot = robot
        robot.world.request_nav_memory_map(0.5)
        self.init_pose = robot.pose
        self.queue = deque(maxlen=1)
        self.date_timestamp = date_timestamp
        self.save_data = save_data
        self.experiment_shorthand_name = ExperimentName(experiment_name).name
        self.experiment_name = experiment_name
        self.execution_uuid = execution_uuid
        self.max_turn_count = max_turn_count
        self.manual_control = manual_control

        self.rand_seed = rand_seed if rand_seed is not None else np.random.randint(100000)
        print(f"Random seed is {self.rand_seed}")

        if experiment_name == ExperimentName.a.value:  # without CLIP all we have is a T/F binary flag for if an obj is detected or not
            self.sensory_space_size = 1  # T or False binary value
        else:  # including CLIP
            # self.sensory_space_size = 384  #DINO SENSORY SPACE# TODO: add location info will consist of obj relative location + size feats concat w DINO output
            self.sensory_space_size = 519  #CLIP+ SENSORY SPACE

        if agent is None:
            print(f"Starting new execution with uuid {self.execution_uuid} and date {self.date_timestamp}")
            # ## 18 x 12 in exploration space
            # m_mins = [-500, -300, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [500, 300, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING
            #
            # ## 18 x 18 in exploration space
            # m_mins = [-500, -500, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [500, 500, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING

            #  ~5 x 5 inch (+/-) in exploration space
            m_mins = [-100, -100, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            m_maxs = [100, 100, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING


            s_mins = [-1] * self.sensory_space_size  # -1 because 0 is a valid CLIP output
            s_maxs = [1] * self.sensory_space_size

            self.cozmo_env = CozmoEnvironment(
                cozmo_robot=robot,
                m_mins=m_mins,
                m_maxs=m_maxs,
                s_mins=s_mins,
                s_maxs=s_maxs,
            )
            self.sensorimotor_model = SensorimotorModel.from_configuration(self.cozmo_env.conf, 'LWLR-NONE', 'default')
            # self.sensorimotor_model = SensorimotorModel.from_configuration(self.cozmo_env.conf, 'NSLWLR-NONE', 'default')
            # Select Interest Model config based on Experiment
            config_name = experiment_name
            self.interest_model = InterestModel.from_configuration(self.cozmo_env.conf, self.cozmo_env.conf.m_dims, 'tree', config_name, rand_seed=self.rand_seed) # passing nav mem map here because we rely on pass by reference for dynamic updates.
            self.agent = ReticoAgent(self.cozmo_env.conf, self.sensorimotor_model, self.interest_model, execution_uuid=self.execution_uuid, execution_date_timestamp=self.date_timestamp, save_data=self.save_data, experiment_name=experiment_name, rand_seed=self.rand_seed)  # agent is necessary to avoid bootstrapping issues

        else:
            self.agent = agent
            print(f"Loading prior execution with uuid {self.execution_uuid} and date {self.date_timestamp}. Continuing with experiment '{experiment_name}'")
            self.cozmo_env = CozmoEnvironment(
                cozmo_robot=robot,
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
            time.sleep(0.05)
            if len(self.queue) == 0:
                continue

            else:
                input_iu = self.queue.popleft()
                self.time_slept = 0
            # when new objects are observed (i.e., not SpeechRecognitionIUs)
            if isinstance(input_iu, ObjectPermanenceIU):
                output_iu = self.create_iu(grounded_in=input_iu)
                motor_action = input_iu.motor_action
                object_features = input_iu.grounded_in.grounded_in.payload #object features
                # If an object was detected but too far away, object features would have data but the Object Detection payload should be empty
                if len(input_iu.payload) == 0:
                    print("Didn't get feature, setting to -1 and continuing.")
                    sensori_effect = [-1]*self.sensory_space_size
                    label = 'whitespace'
                else:
                    if self.sensory_space_size == 1: # ignore the CLIP output and flag as "1" for obj detected
                        sensori_effect = [1]
                    else:
                        sensori_effect = object_features[0][0] # object features
                    # If at a future point we care what YOLO thought it was, then pass that through and access using input_iu.grounded_in.grounded_in
                    # or pass it along
                    label = 'something'
                    logger.log(logging.INFO, f"Something is {input_iu.grounded_in.payload['distance_mm']}mm away")

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

                    # Using pickle instead of csv because I need the objects for easier rendering with the existing opengl implementation.
                    offline_data_path = f'./IAC_output_data/{self.date_timestamp}/{self.execution_uuid}'
                    Path(offline_data_path).mkdir(parents=True, exist_ok=True)
                    with open(f'{offline_data_path}/motor_actions_{self.execution_uuid}.pickle', 'ab+') as file_handler:
                        pickle.dump(self.robot.pose, file_handler)

                    with open(f'{offline_data_path}/nav_memory_map_snapshots_{self.execution_uuid}.pickle', 'ab+') as file_handler:
                        pickle.dump(self.robot.world.nav_memory_map, file_handler)

                    with open(f'{offline_data_path}/seen_objects_{self.execution_uuid}.pickle', 'ab+') as file_handler:
                        pickle.dump(list(copy.deepcopy(self.robot.world._objects).values()), file_handler)

                    with open(f'{offline_data_path}/camera_view_{self.execution_uuid}.pickle', 'ab+') as file_handler:
                        img_bbox = input_iu.grounded_in.grounded_in.grounded_in.image_bbox
                        if img_bbox:
                            draw = ImageDraw.Draw(input_iu.grounded_in.grounded_in.grounded_in.image) #this impacts the input iu image but I don't think we use it again so it's fine.
                            draw.rectangle(((img_bbox['x1'], img_bbox['y1']), (img_bbox['x2'], img_bbox['y2'])), fill=None, outline='green')

                        pickle.dump(input_iu.grounded_in.grounded_in.grounded_in.image, file_handler)



                # inform the agent of the sensorimotor consequence of the action and update both the sensorimotor and interest models
                self.agent.perceive(sensori_effect, flow_uuid=input_iu.flow_uuid, nav_memory_map=self.robot.world.nav_memory_map)
                self.robot.camera.image_stream_enabled = True  # image stream is disabled in retico camera extractor (don't want images when turning)
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
                # run without manual motor goal no matter what so we can print what explauto _would have chosen_ (produce has logging in function so
                # the inference/chosen motor action will be printed without further calls here)
                motor_goal = self.agent.produce(flow_uuid=flow_uuid4)

                if self.manual_control:
                    logger.log(logging.INFO, "Cozmo is ready to drive")
                    # logger.log(logging.INFO, f"[{self.execution_uuid}] Explauto motor goal: {self.agent.x}")
                    # logger.log(logging.INFO, f"[{self.execution_uuid}] Explauto inference: {list(self.agent.y)}") # pass as a list so it doesn't wrap

                    time.sleep(5)
                    starting_pose = self.robot.pose.position.x_y_z
                    while True:
                        prior_pose = self.robot.pose
                        time.sleep(5)
                        # If the robot hasn't moved _at all_ don't break out of loop but if it has moved *and* is no longer moving, then break
                        if starting_pose != self.robot.pose.position.x_y_z and self.robot.pose.position.x_y_z == prior_pose.position.x_y_z:
                            break
                    robot_pose = self.robot.pose
                    # overwrite explauto picked motor goal with the manual one we picked
                    motor_goal = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.rotation.angle_z.degrees])
                    logger.log(logging.INFO, f"Using manual pose: {motor_goal}")
                    self.agent.produce(flow_uuid=flow_uuid4, manual_choice=motor_goal)
                else:
                    # TODO: We don't get sensorimotor impact in this update in the way explauto expects, because we run clip processing as a separate
                    #  retico IU. All we do is move to the motor goal.
                    self.cozmo_env.update(motor_goal, log=False)

                output_iu.set_motor_action(motor_action=motor_goal, flow_uuid=flow_uuid4, execution_uuid=self.execution_uuid)
                um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                self.append(um)
                self.num_ius_processed += 1

    def prepare_run(self):
        # `produce` calls `sample()` method on interest model then uses the sensorimotor model to obtain sensorimotor
        # vector using forward prediction (if motor babbling). Returns the motor part of the full sensorimotor vector.
        flow_uuid4 = str(uuid.uuid4()).split("-")[0]
        # initial motor goal is just where the robot is, this is because we need to perceive to get our first nav_memory_map to sample from
        motor_goal = np.array([self.robot.pose.position.x, self.robot.pose.position.y, self.robot.pose.rotation.angle_z.degrees])
        self.agent.produce(flow_uuid=flow_uuid4, manual_choice=motor_goal)

        if self.manual_control:
            logger.log(logging.INFO, "Cozmo is ready to drive")
            # logger.log(logging.INFO, f"[{self.execution_uuid}] Explauto motor goal: {self.agent.x}")
            # logger.log(logging.INFO, f"[{self.execution_uuid}] Explauto inference: {list(self.agent.y)}") # pass as a list so it doesn't wrap

            time.sleep(5)
            starting_pose = self.robot.pose.position.x_y_z
            while True:
                prior_pose = self.robot.pose
                time.sleep(5)
                # If the robot hasn't moved _at all_ don't break out of loop but if it has moved *and* is no longer moving, then break
                if starting_pose != self.robot.pose.position.x_y_z and self.robot.pose.position.x_y_z == prior_pose.position.x_y_z:
                    break
            robot_pose = self.robot.pose
            # overwrite explauto picked motor goal with the manual one we picked
            motor_goal = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.rotation.angle_z.degrees])
            logger.log(logging.INFO, f"Using manual pose: {motor_goal}")
            # this sets up the inference  as well, so just pass our motor goal instead of choosing
            self.agent.produce(flow_uuid=flow_uuid4, manual_choice=motor_goal)

        else:
            # Execute the motor goal. We cannot get the sensori effect yet.
            self.cozmo_env.update(motor_goal, log=False)

        output_iu = self.create_iu(grounded_in=None)
        output_iu.set_motor_action(motor_action=motor_goal, flow_uuid=flow_uuid4, execution_uuid=self.execution_uuid)
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False
