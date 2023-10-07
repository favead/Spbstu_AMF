"""
Инцициализация модели
"""
from typing import Any, Dict

import lightning as L


def build_model(model_config: Dict[str, Any]) -> L.LightningModule | None:
    """
    Сборка модели на основе передаваемых параметров
    и названии класса модели
    """
    model_name = model_config.get("name", None)
    return None
