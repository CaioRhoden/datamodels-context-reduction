from dataclasses import dataclass

@dataclass
class NaiveSetterConfig:
    load_path: str
    save_path: str
    k: int
    n_samples: int
    test_samples: int

@dataclass
class StratifiedSetterConfig:
    load_path_target: str
    load_path_random: str
    save_path: str
    k: int
    n_samples_target: int
    n_test_target: int
    n_samples_mix: int
    n_test_mix: int
    n_samples_random: int
    n_test_random: int
    index_col: str
    seed: int
    