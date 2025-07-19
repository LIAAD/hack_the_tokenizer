import json
import numpy as np
import torch
import gzip

class TensorJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder that handles PyTorch Tensors, NumPy arrays, and other non-serializable objects."""
    
    def default(self, obj):
        # PyTorch Tensors → Python list
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()  # Move to CPU and convert to list
        
        # NumPy arrays → Python list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle other non-serializable objects (e.g., datetime)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Fallback to string representation

def dump_json(obj: dict, directory: str, use_gzip: bool=False):
    if use_gzip:
        if not directory.endswith('gz'): directory += '.gz'
        with gzip.open(directory, 'wt') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
        return True
    with open(directory, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
    return True
    