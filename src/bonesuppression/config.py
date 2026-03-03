from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class DataConfig:
    source_dir: str = "Dataset/source"
    target_dir: str = "Dataset/target"
    test_dir: Optional[str] = None
    image_size: int = 512
    channels: int = 1
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass
class ModelConfig:
    name: str = "autoencoder"
    base_filters: int = 32
    loss: str = "mixed_l2"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    seed: int = 42
    shuffle_buffer: int = 512


@dataclass
class ExperimentConfig:
    output_dir: str = "outputs"
    experiment_name: str = "baseline"
    resume_checkpoint: Optional[str] = None


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _read_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs. Use JSON config or install PyYAML.") from exc
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    data: Dict[str, Any] = {}
    if config_path:
        data = _read_config_file(config_path) or {}
    if overrides:
        data = _deep_update(data, overrides)

    return AppConfig(
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        train=TrainConfig(**data.get("train", {})),
        experiment=ExperimentConfig(**data.get("experiment", {})),
    )


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    return asdict(config)


def save_config(config: AppConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_dict(config), f, sort_keys=False)
    except ImportError:
        with open(output_path.with_suffix('.json'), "w", encoding="utf-8") as f:
            json.dump(config_to_dict(config), f, indent=2)


def save_config_json(config: AppConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2)
