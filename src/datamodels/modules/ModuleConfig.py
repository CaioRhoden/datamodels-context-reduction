from dataclasses import dataclass
from typing import Optional
from src.datamodels.config import LogConfig


@dataclass
class ModuleConfig:
    """
    log: indicates if the module should be logged in wandb
    log_config: configuration for the logging (see LogConfig)
    """
    log: bool
    log_config: Optional[LogConfig]

@dataclass
class CollectionSetterConfig(ModuleConfig):
    pass

@dataclass
class PreCollectionsConfig(ModuleConfig):
    type: str
    start_idx: int
    end_idx: int
    checkpoint: int
    input_column: str
    output_column: str
    optional_output_column: Optional[str]

@dataclass
class CollectionsConfig(ModuleConfig):
    pass

@dataclass
class TrainerConfig(ModuleConfig):
    pass



