from dataclasses import dataclass


@dataclass
class BaselineConfig:
    llm_type: str
    dataset: str
    instructions: str
    saving_path: str
    run_instructions: bool
    start: int

