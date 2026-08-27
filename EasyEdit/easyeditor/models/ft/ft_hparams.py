from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str

    # Defaults
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False
    full_ft: bool = False
    ft_loss_objective: str = "standard_ft"
    target_alpha: float = 0.1
    pl_skip_above_target: bool = False
    skip_prob: float = 1.0
    rewrite_kl_lambda: float = 0.0
    append_eos_to_target: bool = False
    bf16: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'FT') or print(f'FTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
