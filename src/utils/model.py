# propagate_and_score/src/utils/model.py

import importlib
import torch

def try_import(module_name: str, class_name: str):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except Exception:
        return None

def load_unet_arch(model_type: str, in_ch: int, out_ch: int):
    if model_type == 'unet3d':
        UnetCls = try_import('train.models.Unet3D', 'Unet3D')
    elif model_type == 'unet3d_larger_skip':
        UnetCls = try_import('train.models.Unet3D_larger_skip', 'Unet3D_larger_skip')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if UnetCls is None:
        raise ImportError(f"Could not import {model_type} from 'train.models'. "
                          "Please ensure the module is correctly defined and available.")
    
    return UnetCls(in_ch, out_ch)

def load_weights(model, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)} keys")
    model.to(device).eval()
    return model

def load_unet(checkpoint_path: str, model_type: str, device: str='cpu', in_ch: int=3, out_ch: int=1) -> torch.nn.Module:
    model = load_unet_arch(model_type, in_ch, out_ch)
    return load_weights(model, checkpoint_path, device)