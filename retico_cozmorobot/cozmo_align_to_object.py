"""
Align to Object Module
==================

This module aligns Cozmo to an object. Similar to behaviour found in retico_cozmorobot/cozmo_behaviors
"""

import os
import pickle
import shutil
import threading
import time
from collections import deque
from pathlib import Path


from cozmo.util import degrees, Angle
from opentelemetry import trace

import retico_core
from retico_core.helper_funcs import get_first_instance_of_target_grounded_iu
from retico_vision.vision import ObjectFeaturesIU, ObjectPermanenceIU

tracer = trace.get_tracer("my.tracer.name")

class CozmoAlignToObject(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Cozmo align to object"

    @staticmethod
    def description():
        return "A module that moves Cozmo to face an object"

    @staticmethod
    def input_ius():
        return [ObjectPermanenceIU] # Object permanence IU for adding server side object to nav map on client side

    @staticmethod
    def output_iu():
        return None

    def __init__(self, robot, **kwargs):
        super().__init__(**kwargs)

        self.robot = robot
        self.robot_fov = Angle(self.robot.camera.config.fov_x.radians).degrees
        self.queue = deque(maxlen=1)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    @tracer.start_as_current_span("align_to_object_find_coordinates")
    def find_coordinates(self, go_box):
        img_width = go_box['input_img_w']
        xmin = go_box['x1']/img_width # scale to 0-1
        xmax = go_box['x2']/img_width # scale to 0-1
        obj_width = xmax - xmin
        # getting the center of the obj bounding box
        x_obj_center = (obj_width/2) + xmin
        # if outside range to consider centered
        if x_obj_center > 0.56 or x_obj_center < 0.44:
            x_diff = x_obj_center - 0.5
            turn_angle = (x_diff * (-self.robot_fov)) / 2
        else:
            turn_angle = 0

        return turn_angle

    @tracer.start_as_current_span("align_to_object_turn_toward_object")
    def turn_toward_object(self, object_bbox):
        turn_angle = self.find_coordinates(object_bbox)
        self.robot.turn_in_place(degrees(turn_angle), in_parallel=True).wait_for_completed()
        return turn_angle

    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue
            with tracer.start_as_current_span("align_to_object_extractor") as span:
                input_iu = self.queue.popleft()
                span.add_event("after queue pop")
                flow_uuid = input_iu.meta_data.get('flow_uuid')
                # If the robot instance is accessible, add the object to the nav map.
                # TODO do I want to move away from get() so it fails if missing?
                execution_uuid = input_iu.meta_data.get('execution_uuid')
                date_timestamp = input_iu.meta_data.get('date_timestamp')
                save_data = input_iu.meta_data.get('save_data')
                prior_execution_date_timestamp = input_iu.meta_data.get('prior_execution_date_timestamp')

                if len(input_iu.payload) != 0:
                    print(f"[{flow_uuid}] Moving Cozmo to face object [{flow_uuid}]")
                    # because emotion from GRED might include robot movement (spinning) and we execute the two in parallel, we can't use the current robot pose
                    grounded_object_features_iu = get_first_instance_of_target_grounded_iu(input_iu, [ObjectFeaturesIU])
                    self.turn_toward_object(grounded_object_features_iu.image_bbox)

                if save_data:
                    with tracer.start_as_current_span("align_to_object_save_data") as span:
                        offline_data_dir = f'./IAC_output_data/{date_timestamp}/data_for_offline_replay/{execution_uuid}'

                        if not Path(offline_data_dir).is_dir():
                            Path(offline_data_dir).mkdir(parents=True, exist_ok=True)
                        filename = f"poses_after_centering_to_objects_{execution_uuid}.pickle"
                        split_execution_uuid = execution_uuid.split("_")
                        # only want to copy and rename prior execution files if running from a prior execution and if it hasn't been copied already
                        if len(split_execution_uuid) > 1:
                            if not os.path.exists(f"{offline_data_dir}/{filename}"):
                                prior_execution_uuid = "_".join(split_execution_uuid[0:-1])
                                print(f"[{flow_uuid}] Copying prior execution ({prior_execution_uuid}) data to new execution ({execution_uuid}) directory")
                                prior_execution_dir = f'./IAC_output_data/{prior_execution_date_timestamp}/data_for_offline_replay/{prior_execution_uuid}'
                                prior_execution_filename =  f"poses_after_centering_to_objects_{prior_execution_uuid}.pickle"
                                shutil.copyfile(f'{prior_execution_dir}/{prior_execution_filename}',
                                                    f'{offline_data_dir}/{filename}')

                        with open(f'{offline_data_dir}/poses_after_centering_to_objects_{execution_uuid}.pickle', 'ab+') as file_handler:
                            pickle.dump(self.robot.pose, file_handler)


    def prepare_run(self):
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False
