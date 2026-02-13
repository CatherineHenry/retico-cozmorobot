# retico
import pickle
from pathlib import Path

import retico_core
from retico_vision import ObjectPermanenceIU
from retico_vision.vision import CozmoNavigationMemoryMapIU


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

    def process_update(self, update_message):
        for input_iu, update_type in update_message:
            if update_type != retico_core.UpdateType.ADD:
                continue

        # Save data if IU type is ObjectPermanenceIU (not IACInitializationIU)
        # This ensures we have the date timestamp and execution uuid from server (in case agent was loaded)
        if isinstance(input_iu, ObjectPermanenceIU):
            execution_uuid = input_iu.meta_data['execution_uuid']
            date_timestamp = input_iu.meta_data['date_timestamp']
            offline_data_path = f'./IAC_output_data/{date_timestamp}/{execution_uuid}'
            Path(offline_data_path).mkdir(parents=True, exist_ok=True)
            nav_mem_map = self.robot.world.nav_memory_map
            with open(f'{offline_data_path}/nav_memory_map_snapshots_{execution_uuid}.pickle', 'ab+') as file_handler:
                pickle.dump(nav_mem_map, file_handler)

        output_iu = self.create_iu(input_iu)
        output_iu.set_payload(nav_mem_map)
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

