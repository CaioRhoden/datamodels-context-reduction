from dataclasses import dataclass
from typing import Optional
from src.datamodels.config import LogConfig


@dataclass
class HandlerConfig:
    log: bool
    log_config: Optional[LogConfig]

@dataclass
class CollectionSetterConfig(HandlerConfig):
    pass

@dataclass
class PreCollectionsConfig(HandlerConfig):
    pass

@dataclass
class CollectionsConfig(HandlerConfig):
    pass

@dataclass
class TrainerConfig(HandlerConfig):
    pass



