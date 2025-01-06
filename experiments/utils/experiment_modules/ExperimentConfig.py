from dataclasses import dataclass
from typing import List

@dataclass
class DatamodelSchemeConfig:
    train_file: str
    test_file: str
    tasks: int | List[str] | str
    train_samples: int
    test_samples: int
    num_train_collections: int
    num_test_collections: int

