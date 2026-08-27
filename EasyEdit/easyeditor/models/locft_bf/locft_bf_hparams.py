from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class LocFTBFHyperParams:
    alg_name: str
    device: int
    model_name: str
    layers: List[int]
    rewrite_module_tmp: str
    num_steps: int
    lr: float
    weight_decay: float

    model_name_or_path: Optional[str] = None
    data_path: Optional[str] = None
    save_model_dir: Optional[str] = None
    easyedit_path: Optional[str] = None
    sample_size: Optional[int] = None
    seed: int = 42
    objective: str = "overtone"
    overtone_lambda: float = 0.1
    overtone_epsilon: float = 0.01
    overtone_nsigma: float = 0.5
    target_kl_lambda: float = 0.0
    target_kl_direction: str = "current_to_ref"
    save_tag: str = "baseline"
    dataset_name: Optional[str] = None

    target_alpha: float = 0.1
    batch_size: int = 64
    bf16: bool = False
    model_parallel: bool = False
    early_stop_loss: float = 1e-2
    low_loss_skip_threshold: float = 1e-2

    # Kept for CLI bookkeeping and output naming compatibility.
    break_loss: float = None
    break_prob: float = None
    clamp_norm_factor: float = None

    def __post_init__(self):
        if self.model_name_or_path is None:
            self.model_name_or_path = self.model_name
        self.layers = list(self.layers)

    @staticmethod
    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    config[key] = float(value)
                except ValueError:
                    pass
        return config

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            config = cls.construct_float_from_scientific_notation(config)

        if config and "model_name" not in config and "model_name_or_path" in config:
            config["model_name"] = config["model_name_or_path"]

        if not config or config.get("alg_name") not in {"LocFT-BF", "KLEdit"}:
            alg_name = config.get("alg_name") if config else None
            raise ValueError(
                f"LocFTBFHyperParams can not load from {hparams_name_or_path}, "
                f"alg_name is {alg_name}"
            )
        return cls(**config)
