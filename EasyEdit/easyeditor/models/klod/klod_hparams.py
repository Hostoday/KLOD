from dataclasses import dataclass
from typing import List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class KLODHyperParams(HyperParams):
    alg_name: str
    device: int
    model_name: str

    layers: List[int]
    rewrite_module_tmp: str

    num_steps: int
    lr: float
    weight_decay: float
    target_alpha: float

    batch_size: int = 64
    bf16: bool = False
    model_parallel: bool = False
    kl_lambda: float = 0.0
    model_name_or_path: Optional[str] = None

    data_path: Optional[str] = None
    save_model_dir: Optional[str] = None
    easyedit_path: Optional[str] = None
    rewrite_kl_lambda: Optional[float] = None
    non_target_kl_lambda: float = 0.0
    sample_size: Optional[int] = None
    seed: int = 42
    early_stop_loss: Optional[float] = None
    use_context_aug: int = 0
    context_aug_stage: str = "klod"
    context_aug_length_params: str = "10:5"
    context_aug_max_templates: Optional[int] = None
    save_tag: str = "KLOD"
    dataset_name: Optional[str] = None

    # Legacy fields kept so older configs with extra EasyEdit keys still load.
    stats_dir: str = "./data/stats"
    edit_layers: List[int] = None
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 25
    v_lr: float = 5e-1
    v_loss_layer: int = 0
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 1.0
    kl_factor: float = 0.0
    context_template_length_params: List[List[int]] = None
    layer_module_tmp: str = "model.layers.{}"
    mlp_module_tmp: str = "model.layers.{}.mlp"
    attn_module_tmp: str = "model.layers.{}.self_attn"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"
    norm_constraint: float = False
    objective_optimization: str = "target_new"
    full_ft: bool = False
    max_length: int = 40
    loss_fn: str = "diff"
    break_loss: float = 0.1
    neuron_k: float = 1.0
    mse_lambda: float = 0.1
    break_prob: float = None

    def __post_init__(self):
        if self.model_name_or_path is None:
            self.model_name_or_path = self.model_name
        if self.edit_layers is None:
            self.edit_layers = list(self.layers)
        if self.context_template_length_params is None:
            self.context_template_length_params = "None"
        if self.v_loss_layer == 0 and self.layers:
            self.v_loss_layer = max(self.layers)

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        if config and "model_name" not in config and "model_name_or_path" in config:
            config["model_name"] = config["model_name_or_path"]

        if not config or config.get("alg_name") not in {"KLOD", "KLEdit"}:
            alg_name = config.get("alg_name") if config else None
            raise ValueError(
                f"KLODHyperParams can not load from {hparams_name_or_path}, "
                f"alg_name is {alg_name}"
            )
        return cls(**config)
