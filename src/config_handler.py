"""
Handles configuration settings and allows other scripts to read the config.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from yaml import safe_load as yaml_safe_load

class BoardConfig(BaseModel):
    starting_position: List[List[Optional[str]]] = Field(
        default_factory=lambda: [[None for _ in range(8)] for _ in range(8)],
        description="Initial board configuration with None/null for empty cells."
    )

class Config(BaseModel):
    board: BoardConfig

def get_config(file_path: str) -> Config:
    """
    Reads the configuration from a YAML file and returns a Config object.
    """
    with open(file_path, 'r') as file:
        config_data = yaml_safe_load(file)
    return Config(**config_data)