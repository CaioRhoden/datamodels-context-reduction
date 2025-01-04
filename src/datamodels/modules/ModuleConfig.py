from dataclasses import dataclass
from typing import Optional
from src.datamodels.config import LogConfig


@dataclass
class ModuleConfig:
    log: bool
    log_config: Optional[LogConfig]

@dataclass
class CollectionSetterConfig(ModuleConfig):
    pass

@dataclass
class PreCollectionsConfig(ModuleConfig):
    pass

@dataclass
class CollectionsConfig(ModuleConfig):
    pass

@dataclass
class TrainerConfig(ModuleConfig):
    pass



