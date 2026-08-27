from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..dataset import ZsreDataset, CounterFactDataset, CaptionDataset, VQADataset, PersonalityDataset, SafetyDataset
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from ..models.ultraedit import UltraEditHyperParams, UltraEditRewriteExecutor
from ..models.klod import KLODHyperParams


ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "ULTRAEDIT": UltraEditRewriteExecutor().apply_to_model,
}

ALG_MULTIMODAL_DICT = {
}

PER_ALG_DICT = {
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}

PER_DS_DICT = {
    "personalityEdit": PersonalityDataset
}
Safety_DS_DICT ={
    "safeEdit": SafetyDataset
}
