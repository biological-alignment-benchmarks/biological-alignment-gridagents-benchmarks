from pathlib import Path

from omegaconf import OmegaConf


def get_project_path(path_from_root: str) -> Path:
    project_root = Path(__file__).parents[2]
    return project_root / path_from_root


def register_resolvers() -> None:
    OmegaConf.register_resolver("abs_path", get_project_path)
