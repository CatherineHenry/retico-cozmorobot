import logging
import pickle
import sys
import tkinter as tk
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from explauto import InterestModel, SensorimotorModel
from explauto.agent import ReticoAgent
from explauto.environment.cozmo_env import CozmoEnvironment

import retico_core
from retico_core import abstract, UpdateType
from retico_core.helper_funcs import get_first_instance_of_target_grounded_iu
from retico_core.robot import IACMotorGoalIU, RobotStateIU
from retico_cozmorobot.initialize_cozmo_IAC import IACInitializationIU
from retico_vision import CozmoNavigationMemoryMapIU, ObjectFeaturesIU, ObjectPermanenceIU

import shutil


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
    g = 'cozmo_clip_cos_sim_split_progressive_splits'

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
        return [IACInitializationIU, CozmoNavigationMemoryMapIU]

    @staticmethod
    def output_iu():
        return IACMotorGoalIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # These will be set in a call to setup_iac when we receive the first (and only) IACInitializationIU
        # This is so we only set a configuration in one place, to limit error and simplify the pipeline
        self.save_data = None
        self.experiment_name = None
        self.experiment_shorthand_name = None
        self.experiment_shorthand_name = None
        self.execution_uuid = None
        self.max_turn_count = None
        self.manual_control = None

        # Is set based on if an agent is loaded
        self.agent = None
        self.date_timestamp = None
        self.sensorimotor_model = None
        self.interest_model = None
        self.rand_seed = None

    def setup_iac(self, grounded_motor_action_iu):
        iu_meta_data = grounded_motor_action_iu.meta_data
        self.save_data = iu_meta_data.get('save_data')
        self.max_turn_count = iu_meta_data.get('max_turn_count')
        self.manual_control = iu_meta_data.get('manual_control')
        # If we are loading a prior execution, the execution_uuid will already be set
        self.execution_uuid = iu_meta_data.get('execution_uuid')

        # If an execution ID was included, we are loading a prior execution
        if self.execution_uuid:
            self.date_timestamp = iu_meta_data.get('date_timestamp')

            updated_execution_uuid = f"{self.execution_uuid}_{str(uuid.uuid4()).split('-')[0]}"
            print(f"Updated execution uuid: {updated_execution_uuid}")
            shutil.copyfile(f'./IAC_output_data/{self.date_timestamp}/agent_{self.execution_uuid}.pickle',
                            f'./IAC_output_data/{self.date_timestamp}/agent_{updated_execution_uuid}.pickle')
            shutil.copyfile(f'./IAC_output_data/{self.date_timestamp}/sensori_effect_{self.execution_uuid}.csv',
                            f'./IAC_output_data/{self.date_timestamp}/sensori_effect_{updated_execution_uuid}.csv')

            # TODO: this is currently hardcoded for bb type, update if we end up supporting other configs
            # clip_bb_path = Path(f"./extraction_output/{self.date_timestamp}/bb/{updated_execution_uuid}/extracted/")
            # clip_bb_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(f"./extraction_output/{self.date_timestamp}/bb/{self.execution_uuid}/extracted/",
                            f"./extraction_output/{self.date_timestamp}/bb/{updated_execution_uuid}/extracted/",
                            dirs_exist_ok = True)

            with open(f'./IAC_output_data/{self.date_timestamp}/agent_{self.execution_uuid}.pickle', 'rb') as f:
                self.agent = pickle.load(f)
            self.execution_uuid = updated_execution_uuid
            overridden_experiment_name = iu_meta_data.get('experiment_name')
            if overridden_experiment_name is not None:
                print(f"Loading prior execution with uuid {self.execution_uuid} and date {self.date_timestamp}. Continuing with different experiment '{overridden_experiment_name}'")
                self.experiment_name = overridden_experiment_name  # override whatever was used in the loaded model with the specified experiment name
                self.experiment_shorthand_name = ExperimentName(self.experiment_name).name
            else:
                print(f"Loading prior execution with uuid {self.execution_uuid} and date {self.date_timestamp}. Continuing with experiment '{self.experiment_name}'")
                self.experiment_name = self.agent.experiment_name  # override experiment with whatever was used in the loaded model
                self.experiment_shorthand_name = ExperimentName(self.experiment_name).name

            self.rand_seed = self.agent.rand_seed
            self.interest_model = self.agent.interest_model
            self.sensorimotor_model = self.agent.sensorimotor_model

        # Starting a fresh execution
        else:
            self.execution_uuid = str(uuid.uuid4()).split("-")[0]
            self.date_timestamp = datetime.now().strftime('%m_%d')
            print(f"Starting new execution with uuid {self.execution_uuid} and date {self.date_timestamp}")
            self.experiment_name = iu_meta_data['experiment_name'] # TODO: move to if an agent was not loaded
            self.experiment_shorthand_name = ExperimentName(self.experiment_name).name
            self.rand_seed = np.random.randint(100000)

            if self.experiment_name == ExperimentName.a.value:
                # without CLIP all we have is a T/F binary flag for if an obj is detected or not
                sensory_space_size = 1  # T or False binary value
            else:  # including CLIP
                sensory_space_size = 519  #CLIP+ SENSORY SPACE




            # -5in | +18in X, +-12in Y  exploration space
            m_mins = [-127, -300, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            m_maxs = [482, 300, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING


            # ## 18 x 12 in exploration space *2
            # m_mins = [-500, -300, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [500, 300, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING
            #
            #
            # # 12 x 12 exploration space
            # m_mins = [-300, -300, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [300, 300, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING


            # # ## 18 x 18 in exploration space
            # m_mins = [-500, -500, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [500, 500, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING

            #  ~5 x 5 inch (+/-) in exploration space
            # m_mins = [-100, -100, -180]   # Cozmo Pose x,y (width and length of space + rotation) distance in mm # SOME PADDING
            # m_maxs = [100, 100, 180]  # Cozmo Pose x,y + rotation distance in mm # SOME PADDING

            s_mins = [-1] * sensory_space_size  # -1 because 0 is a valid CLIP output
            s_maxs = [1] * sensory_space_size

            # cozmo_env is init both server _and_ client side, the client side will have access to the robot instance
            # to perform movements. **Only used for conf value on server!**. We pass the values to init from server so
            # they are guaranteed using the same data.
            # We use the env a little differently from original Explauto design, since we use the cozmo_camera module
            # to collect and send the sensory output (camera feed)
            cozmo_env = CozmoEnvironment(
                cozmo_robot=None,
                m_mins=m_mins,
                m_maxs=m_maxs,
                s_mins=s_mins,
                s_maxs=s_maxs,
            )
            self.sensorimotor_model = SensorimotorModel.from_configuration(cozmo_env.conf, 'LWLR-NONE', 'default')
            # self.sensorimotor_model = SensorimotorModel.from_configuration(self.cozmo_env.conf, 'NSLWLR-NONE', 'default')

            # Select Interest Model config based on Experiment
            config_name = self.experiment_name
            self.interest_model = InterestModel.from_configuration(cozmo_env.conf, cozmo_env.conf.m_dims, 'tree', config_name, rand_seed=self.rand_seed, max_turn_count=self.max_turn_count) # passing nav mem map here because we rely on pass by reference for dynamic updates.
            self.agent = ReticoAgent(cozmo_env.conf, self.sensorimotor_model, self.interest_model, execution_uuid=self.execution_uuid, execution_date_timestamp=self.date_timestamp, save_data=self.save_data, experiment_name=self.experiment_name, rand_seed=self.rand_seed)  # agent is necessary to avoid bootstrapping issues


        Path(f"IAC_output_data/{self.date_timestamp}").mkdir(parents=True, exist_ok=True)
        print(f"Random seed is {self.rand_seed}")

    def process_update(self, update_message):
        for input_iu, update_type in update_message:
            if update_type != UpdateType.ADD:
                continue
        output_iu = self.create_iu(grounded_in=input_iu)
        # read flow_uuid from input_iu and use to complete the remaining perception steps now that we have the sensory
        # data available.
        # Will make a new a flow_uuid at the end when we produce the next motor action
        flow_uuid = input_iu.meta_data.get('flow_uuid')
        # could go by flow ID, but we need the init IU anyway to pass the payload forward
        if isinstance(input_iu, IACInitializationIU):
            self.setup_iac(input_iu)
            output_iu.meta_data['date_timestamp'] = self.date_timestamp
            # If the execution_uuid was None when we started, we need to set it to the new UUID
            # If one was set to run a prior execution this will just overwrite with the same execution_uuid as before
            output_iu.meta_data['execution_uuid'] = self.execution_uuid
            output_iu.meta_data['init_cozmo_env'] = {
                'm_mins': self.sensorimotor_model.conf.m_mins,
                'm_maxs': self.sensorimotor_model.conf.m_maxs,
                's_mins': self.sensorimotor_model.conf.s_mins,
                's_maxs': self.sensorimotor_model.conf.s_maxs
            }
            # TODO: should I pass the input as output for this first run so it is run through the entire pipeline
            motor_goal = input_iu.payload
            # We kick off execution with the robot's initial state as the motor goal
            # Because of the break in movement/perception due to our perception being tied to an IU output (cozmo cam)
            # and sending data from client/server, we cannot have the agent produce an action _and_ perceive in the same
            # run, there would be no updated sensory data to perceive.
            # Calling produce is still important here as it sets variables we use downstream.
            self.agent.produce(flow_uuid=flow_uuid, manual_choice=motor_goal)

        else:
            grounded_motor_action_iu = get_first_instance_of_target_grounded_iu(input_iu, [RobotStateIU])
            # TODO: does this work here?
            if self.manual_control:
                # See what the model would have predicted for the manual motor action
                self.agent.produce(flow_uuid=flow_uuid, manual_choice=grounded_motor_action_iu.payload)

            grounded_object_features_iu = get_first_instance_of_target_grounded_iu(input_iu, [ObjectFeaturesIU]) #object features
            grounded_object_permanence_iu = get_first_instance_of_target_grounded_iu(input_iu, [ObjectPermanenceIU]) #object permanence
            # If an object was detected but too far away, object features would have data but the Object Detection payload should be empty
            if len(grounded_object_permanence_iu.payload) == 0:
                print("Didn't get feature, setting to -1 and continuing.")
                sensori_effect = [-1]*self.sensorimotor_model.conf.s_ndims
                label = 'whitespace'
            else:
                if self.sensorimotor_model.conf.s_ndims == 1: # ignore the CLIP output and flag as "1" for obj detected
                    sensori_effect = [1]
                else:
                    sensori_effect = grounded_object_features_iu.payload[0][0] # object features
                # If at a future point we care what YOLO thought it was, then pass that through and access using input_iu.grounded_in.grounded_in
                # or pass it along
                label = 'something'
                logger.log(logging.INFO, f"Something is {grounded_object_permanence_iu.payload['distance_mm']}mm away")

            inferred_sensori = self.agent.y
            if self.save_data:
                sensori_df = pd.DataFrame.from_records([np.hstack([grounded_motor_action_iu.payload, sensori_effect]), np.hstack([grounded_motor_action_iu.payload, inferred_sensori])])
                sensori_df.insert(0, 'expl_dims', [len(self.agent.expl_dims)]*2)
                sensori_df.insert(0, 'inf_dims', [len(self.agent.inf_dims)]*2)
                sensori_df.insert(0, 'experiment_name', self.experiment_name)
                sensori_df.insert(0, 'sensori_type', ['effect', 'inferred'])
                sensori_df.insert(0, 'obj_name', [label]*2)
                sensori_df.insert(0, 'flow_uuid', [flow_uuid]*2)
                sensori_df.insert(0, 'exec_uuid', [self.execution_uuid]*2)

                sensori_df.to_csv(f'./IAC_output_data/{self.date_timestamp}/sensori_effect_{self.execution_uuid}.csv', mode='a', index=False, header=False)

            # inform the agent of the sensorimotor consequence of the action and update both the sensorimotor and interest models
            self.agent.perceive(sensori_effect, flow_uuid=flow_uuid, nav_memory_map=input_iu.payload)

            turn_count = len(self.interest_model.data_x)
            # We've completed max number of turns, save the model and exit
            if self.max_turn_count != 0 and turn_count == self.max_turn_count:
                self.agent.save(f"./IAC_output_data/{self.date_timestamp}/agent_{self.execution_uuid}.pickle")
                print(f"Successfully ran {self.max_turn_count} actions. Saved agent and quitting program.")
                sys.exit()

            # set new flow uuid for the new motor action
            flow_uuid = str(uuid.uuid4()).split("-")[0]
            motor_goal = self.agent.produce(flow_uuid=flow_uuid)

        output_iu.set_payload(motor_goal=motor_goal)
        output_iu.meta_data['flow_uuid'] = flow_uuid

        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)