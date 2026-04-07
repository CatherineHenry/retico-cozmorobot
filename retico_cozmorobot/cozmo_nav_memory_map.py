# retico
import os.path
import pickle
import shutil
from pathlib import Path

from opentelemetry import trace

import retico_core
from retico_vision import ObjectPermanenceIU
from retico_vision.vision import CozmoNavigationMemoryMapIU

tracer = trace.get_tracer("my.tracer.name")

class CozmoNavMemoryMapModule(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "Cozmo Navigation Memory Map Module"

    @staticmethod
    def description():
        return "A module that sends Cozmo navigation memory map state"

    @staticmethod
    def input_ius():
        return [ObjectPermanenceIU]

    @staticmethod
    def output_iu():
        return CozmoNavigationMemoryMapIU


    def __init__(self, robot, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.robot.world.request_nav_memory_map(0.5)

    @tracer.start_as_current_span("nav_mem_map_process_update")
    def process_update(self, update_message):
        for input_iu, update_type in update_message:
            if update_type != retico_core.UpdateType.ADD:
                continue

        # Save data if IU type is ObjectPermanenceIU (not IACInitializationIU)
        # This ensures we have the date timestamp and execution uuid from server (in case agent was loaded)
        if isinstance(input_iu, ObjectPermanenceIU):
            execution_uuid = input_iu.meta_data.get('execution_uuid')
            date_timestamp = input_iu.meta_data.get('date_timestamp')
            prior_execution_date_timestamp = input_iu.meta_data.get('prior_execution_date_timestamp')
            save_data = input_iu.meta_data.get('save_data')
            nav_mem_map = self.robot.world.nav_memory_map
            if save_data:
                filename = f'nav_memory_map_snapshots_{execution_uuid}.pickle'
                offline_data_dir = f'./IAC_output_data/{date_timestamp}/data_for_offline_replay/{execution_uuid}'
                if not Path(offline_data_dir).is_dir():
                    Path(offline_data_dir).mkdir(parents=True, exist_ok=True)
                split_execution_uuid = execution_uuid.split("_")
                # only want to copy and rename prior execution files if running from a prior execution and if it hasn't been copied already
                if len(split_execution_uuid) > 1:
                    if not os.path.exists(f"{offline_data_dir}/{filename}"):
                        prior_execution_uuid = "_".join(execution_uuid.split("_")[0:-1])
                        prior_execution_dir = f'./IAC_output_data/{prior_execution_date_timestamp}/data_for_offline_replay/{prior_execution_uuid}'
                        prior_execution_filename =  f"nav_memory_map_snapshots_{prior_execution_uuid}.pickle"
                        shutil.copyfile(f'{prior_execution_dir}/{prior_execution_filename}',
                                        f'{offline_data_dir}/{filename}')

                with open(f'{offline_data_dir}/{filename}', 'ab+') as file_handler:
                    pickle.dump(nav_mem_map, file_handler)

        output_iu = self.create_iu(input_iu)
        output_iu.set_payload(nav_mem_map)
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

