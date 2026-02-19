import os
import json
import time
from typing import Dict, Any

from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be unavailable in some contexts
    torch = None  # type: ignore


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert common ML objects to JSON-serializable types.

    - torch.Tensor -> Python number if scalar else list
    - numpy types/arrays -> Python number/list
    - Path -> str
    - Sets/Tuples -> list
    - Unsupported objects -> str(obj)
    """
    # Fast-path for simple JSON-native types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        # Normalize NaN/Inf which default json dumps may not handle in strict parsers
        if isinstance(obj, float):
            if obj != obj or obj in (float("inf"), float("-inf")):
                return None
        return obj

    # torch.Tensor
    if torch is not None and isinstance(obj, torch.Tensor):  # type: ignore
        if obj.numel() == 1:
            try:
                return obj.item()
            except Exception:
                return float(obj.detach().cpu().reshape(()))
        return obj.detach().cpu().tolist()

    # numpy scalars / arrays
    if np is not None:
        if isinstance(obj, getattr(np, "generic", ())):  # numpy scalar
            try:
                return obj.item()
            except Exception:
                return float(obj)
        if isinstance(obj, getattr(np, "ndarray", ())):
            return obj.tolist()

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # mappings
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}

    # sequences
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    # fallback to string representation
    return str(obj)

class JSONLogger:
    """Logger that saves metrics to JSON files."""
    
    def __init__(self, log_dir: str, filename: str = "metrics.jsonl", append: bool = False):
        """
        Initialize the JSON logger.
        
        Args:
            log_dir: Directory to save the JSON log files
            filename: Name of the JSONL file (default: "metrics.jsonl")
            append: If True, append to existing file; if False, clear existing file
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create metrics file or clear it if it exists (unless append mode)
        if not append:
            with open(self.metrics_file, 'w') as f:
                pass
            print(f"JSONLogger initialized. Metrics will be saved to {self.metrics_file}")
        else:
            # In append mode, just ensure the file exists
            if not os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'w') as f:
                    pass
            print(f"JSONLogger initialized in append mode. Metrics will be appended to {self.metrics_file}")
        
    def log(self, data: Dict[str, Any], step: int):
        """
        Log metrics to a JSON file.
        
        Args:
            data: Dictionary of metrics to log
            step: Current step number
        """
        # Add timestamp and step to the data
        log_entry = {
            "timestamp": time.time(),
            "log_step": int(step),
            "metrics": _to_json_serializable(data),
        }
        
        # Append to the JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
