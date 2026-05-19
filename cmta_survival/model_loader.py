import json
import os
import re
from typing import List, Optional

import torch

import config


CMTA_RESULTS_DIR = config.CMTA_RESULTS_DIR


def _checkpoint_score(path: str) -> float:
    match = re.search(r"model_best_([0-9.]+)_", os.path.basename(path))
    if not match:
        return float("-inf")
    try:
        return float(match.group(1))
    except ValueError:
        return float("-inf")


def find_best_model_path(cancer_type: str, fold: int = 0) -> Optional[str]:
    """Find the CMTA checkpoint for a cancer type."""
    result_dir = config.MODEL_CONFIGS[cancer_type]["result_dir"]

    if not os.path.exists(result_dir):
        print(f"CMTA result directory does not exist: {result_dir}")
        return None

    model_files = [
        os.path.join(result_dir, name)
        for name in os.listdir(result_dir)
        if name.startswith("model_best_") and (name.endswith(".pth") or name.endswith(".pth.tar"))
    ]
    if not model_files:
        for root, _, files in os.walk(result_dir):
            for name in files:
                if name.startswith("model_best_") and (name.endswith(".pth") or name.endswith(".pth.tar")):
                    model_files.append(os.path.join(root, name))
    if not model_files:
        print(f"No CMTA model checkpoint found in {result_dir}")
        return None

    model_files.sort(key=lambda path: (_checkpoint_score(path), path), reverse=True)
    return model_files[0]


def _infer_omic_sizes_from_checkpoint(checkpoint) -> List[int]:
    """Infer CMTA omic input sizes from checkpoint weights."""
    state_dict = checkpoint.get("state_dict", checkpoint)

    omic_sizes = []
    for i in range(6):
        key = f"genomics_fc.{i}.0.0.weight"
        if key in state_dict:
            omic_sizes.append(int(state_dict[key].shape[1]))
        else:
            omic_sizes.append(100)
    return omic_sizes


def _load_preprocessing_metadata(model_path: str, checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("preprocessing"), dict):
        return checkpoint["preprocessing"]

    metadata_path = os.path.join(os.path.dirname(model_path), "preprocessing_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    return {}


def load_cmta_checkpoint(model_path: str, device: str = "cpu") -> tuple:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CMTA checkpoint does not exist: {model_path}")

    print(f"[正在加载CMTA模型] {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "best_score" in checkpoint:
        print(f"[最优性能指标(C-index)] {checkpoint['best_score']:.4f}")
    else:
        print("[最优性能指标(C-index)] N/A")

    inferred_omic_sizes = _infer_omic_sizes_from_checkpoint(checkpoint)
    preprocessing = _load_preprocessing_metadata(model_path, checkpoint)
    print(f"[基因组各维度情况] {inferred_omic_sizes}")
    return checkpoint, inferred_omic_sizes, preprocessing


def load_cmta_model(
    cancer_type: str,
    fold: int = 0,
    omic_sizes: List[int] | None = None,
    fusion: str = "concat",
    model_size: str = "small",
    device: str | None = None,
):
    """Load a CMTA model and its checkpoint."""
    from cmta_survival.network import CMTA

    device = device or "cpu"
    model_path = find_best_model_path(cancer_type, fold)
    if model_path is None:
        raise FileNotFoundError(f"Cannot find CMTA checkpoint for {cancer_type}")

    checkpoint, inferred_omic_sizes, preprocessing = load_cmta_checkpoint(model_path, device)
    if omic_sizes is None or sum(omic_sizes) == 0:
        omic_sizes = preprocessing.get("omic_sizes") or inferred_omic_sizes

    model = CMTA(
        omic_sizes=omic_sizes,
        n_classes=4,
        fusion=fusion,
        model_size=model_size,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, checkpoint, omic_sizes


if __name__ == "__main__":
    print("Testing CMTA model loading...")
    for cancer_type in config.SUPPORTED_CANCER_TYPES:
        try:
            print(f"\n--- {cancer_type} ---")
            model, checkpoint, omic_sizes = load_cmta_model(cancer_type)
            print(f"  Model type: {type(model)}")
            print(f"  Best C-index: {checkpoint.get('best_score', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}")
            print(f"  Omic sizes: {omic_sizes}")
        except Exception as exc:
            print(f"  Failed: {exc}")
