"""
Handles configuration settings and allows other scripts to read the config.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from yaml import safe_load as yaml_safe_load

class BoardConfig(BaseModel):
    _default_starting_position: List[List[Optional[str]]] = [
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, 'W', 'B', None, None, None],
        [None, None, None, 'B', 'W', None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None]
    ]
    starting_position: List[List[Optional[str]]] = Field(
        default_factory=lambda: BoardConfig._default_starting_position,
        description="Initial board configuration with None/null for empty cells."
    )

    @field_validator('starting_position')
    def validate_starting_position(cls, value: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
        if len(value) != 8 or any(len(row) != 8 for row in value):
            raise ValueError("Starting position must be an 8x8 grid.")
        for row in value:
            if any(cell not in (None, 'B', 'W') for cell in row):
                raise ValueError("Cells must be None, 'B', or 'W'.")
        return value

    

class Config(BaseModel):
    board: BoardConfig

def get_config(file_path: str) -> Config:
    """
    Reads the configuration from a YAML file and returns a Config object.
    """
    with open(file_path, 'r') as file:
        config_data = yaml_safe_load(file)
    return Config(**config_data)