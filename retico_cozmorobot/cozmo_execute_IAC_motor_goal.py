# retico
import pickle
import time
from pathlib import Path

import numpy as np
from explauto.environment.cozmo_env import CozmoEnvironment

import retico_core
from retico_core.robot import IACMotorGoalIU, RobotStateIU
from retico_vision import ObjectPermanenceIU
from retico_vision.vision import CozmoNavigationMemoryMapIU
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CozmoExecuteIACMotorGoalModule(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "Cozmo Execute IAC Motor Goal Module"

    @staticmethod
    def description():
        return "A module that executes an IAC motor goal"

    @staticmethod
    def input_ius():
        return [IACMotorGoalIU]

    @staticmethod
    def output_iu():
        # TODO: idk, seems good for now though? Set the completed motor goal as the robot state
        return RobotStateIU


    def __init__(self, robot, manual_control=False, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.cozmo_iac_env = None
        self.manual_control = manual_control

    def process_update(self, update_message):
        for input_iu, update_type in update_message:
            if update_type != retico_core.UpdateType.ADD:
                continue

        output_iu = self.create_iu(input_iu)
        #TODO: check if need to initialize cozmo env.
        if 'init_cozmo_env' in input_iu.meta_data.keys():
            self.cozmo_iac_env = CozmoEnvironment(
                cozmo_robot=self.robot,
                m_mins=input_iu.meta_data['init_cozmo_env']['m_mins'],
                m_maxs=input_iu.meta_data['init_cozmo_env']['m_maxs'],
                s_mins=input_iu.meta_data['init_cozmo_env']['s_mins'],
                s_maxs=input_iu.meta_data['init_cozmo_env']['s_maxs'],
            )
            del output_iu.meta_data['init_cozmo_env']

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
                # TODO: this might be broken now that I've split things up...I no longer call produce with these goals so state isn't updated...
                # TODO: might need to output a different IU + topic for manual motor command...
                # And I can't really call produce because I only have access to that on the server side..


                # overwrite explauto picked motor goal with the manual one we picked
                motor_goal = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.rotation.angle_z.degrees])
                logger.log(logging.INFO, f"Using manual pose: {motor_goal}")
        else:
            # Note: We don't get sensorimotor impact in this update in the way explauto expects, because we run clip processing as a separate
            #  retico IU. All we do is move to the motor goal.
            # Execute the motor goal. We cannot get the sensori effect yet.
            self.cozmo_iac_env.update(input_iu.payload, log=False)

        # Save data if IU type is IACMotorGoalIU (not IACInitializationIU).
        # This ensures we have the date timestamp and execution uuid from server (in case agent was loaded)
        if isinstance(input_iu, IACMotorGoalIU):
            execution_uuid = input_iu.meta_data['execution_uuid']
            date_timestamp = input_iu.meta_data['date_timestamp']
            # Using pickle instead of csv because I need the objects for easier rendering with the existing opengl implementation.
            offline_data_path = f'./IAC_output_data/{date_timestamp}/{execution_uuid}'
            Path(offline_data_path).mkdir(parents=True, exist_ok=True)
            with open(f'{offline_data_path}/motor_actions_{execution_uuid}.pickle', 'ab+') as file_handler:
                # TODO: pass the motor goal instead, will have to update how we run it on robot
                # Saving the robot pose just exacerbates to the rotation error already present
                pickle.dump(self.robot.pose, file_handler)

        # Either pass the goal along as the new robot state, or the resulting pose of any manual movement
        output_iu.set_state(motor_goal)
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

