from typing import Union, List, Tuple
from dynamic_network_architectures.architectures.unet import AttentionUNet
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class nnUNetPlannerAttUNet(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansAttUNet',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        """
        Attention U-Net.
        Everything remains the same, except for the Attention UNet architecture.
        """
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = AttentionUNet

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

if __name__ == '__main__':
    nnUNetPlannerAttUNet(2, 8).plan_experiment()
