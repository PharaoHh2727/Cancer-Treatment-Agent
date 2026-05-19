"""
TransMIL模型封装 - 作为LangChain Tool供LLM调用
"""
import json
import csv
import os
import sys
import torch
import numpy as np
from pathlib import Path
'''
# 添加TransMIL路径
TRANSMIL_DIR = Path(__file__).parent.parent / "transmil_tnm"
sys.path.insert(0, str(TRANSMIL_DIR))'''

DEFAULT_CANCER_TYPE = "BRCA"

from transmil_tnm.TransMIL_TNM import TransMIL_TNM
import config


class TransMILPredictor:
    """TransMIL预测器封装类"""
    
    def __init__(self,  cancer_type: str , force_cpu: bool = True):
        self.cancer_type = cancer_type
        self.model_path = config.TRANSMIL_MODEL_PATH[self.cancer_type]["result_dir"]
        self.device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labels = {
            't': config.T_LABELS,
            'n': config.N_LABELS,
            'm': config.M_LABELS,
            'stage': config.STAGE_LABELS
        }
        
    def load_model(self):
        """加载训练好的模型"""
        if self.model is not None:
            return
            
        print(f"[正在加载TransMIL模型] {self.model_path}")
        
        # 创建模型
        model = TransMIL_TNM(
            t_classes=config.T_CLASSES,
            n_classes=config.N_CLASSES,
            m_classes=config.M_CLASSES,
            stage_classes=config.STAGE_CLASSES
        )
        
        # 加载权重 (处理model.前缀 - DataParallel保存的checkpoint)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 去掉 'model.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # 去掉 'model.'
            else:
                new_state_dict[k] = v
        
        # 加载模型权重
        model.load_state_dict(new_state_dict, strict=False)
        
        # 强制移至CPU (处理模型内部的 .cuda() 硬编码)
        self.model = model.to(self.device)
        self.model.eval()
        print(f"[模型已加载到设备] {self.device}")
 
    def _coerce_features(self, features) -> torch.Tensor:
        if isinstance(features, dict):
            for key in ("features", "data"):
                if key in features:
                    features = features[key]
                    break
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        elif not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(f"TransMIL expects features shaped [n_patches, 1024], got {tuple(features.shape)}")
        return features.float()

    def _format_predictions(self, results: dict) -> dict:
        return {
            'T': {
                'label': self.labels['t'][results['t_hat'].item()],
                'prob': results['t_prob'].detach().cpu().numpy()[0].tolist(),
                'confidence': float(results['t_prob'].max().item())
            },
            'N': {
                'label': self.labels['n'][results['n_hat'].item()],
                'prob': results['n_prob'].detach().cpu().numpy()[0].tolist(),
                'confidence': float(results['n_prob'].max().item())
            },
            'M': {
                'label': self.labels['m'][results['m_hat'].item()],
                'prob': results['m_prob'].detach().cpu().numpy()[0].tolist(),
                'confidence': float(results['m_prob'].max().item())
            },
            'Stage': {
                'label': self.labels['stage'][results['stage_hat'].item()],
                'prob': results['stage_prob'].detach().cpu().numpy()[0].tolist(),
                'confidence': float(results['stage_prob'].max().item())
            }
        }

    def _fold_square_padding(self, values: torch.Tensor, original_count: int, square_count: int) -> torch.Tensor:
        values = values[:square_count]
        folded = values[:original_count].clone()
        pad_count = square_count - original_count
        if pad_count > 0:
            folded[:pad_count] = folded[:pad_count] + values[original_count:square_count]
        return folded

    #选择正负分数
    def _select_signed_scores(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros_like(values)
        if bool(mask.any().item()):
            scores[mask] = values[mask]
        return scores

    #标记正负分数，确定计分模式
    def _signed_gradient_scores(self, values: torch.Tensor) -> tuple[torch.Tensor, str, str, dict[str, int]]:
        values = values.detach()
        positive_mask = values > 0
        negative_mask = values < 0
        positive_count = int(positive_mask.sum().item())
        negative_count = int(negative_mask.sum().item())
        counts = {
            "positive": positive_count,
            "negative": negative_count,
            "zero": int(values.numel()) - positive_count - negative_count,
        }

        if positive_count > 0:
            return (
                self._select_signed_scores(values, positive_mask),
                "positive_relu_raw",
                "\u5b58\u5728\u6b63\u6570patch\u65f6\uff0cattention*gradient\u8d1f\u503c\u7ecfReLU\u7f6e0\uff1bnormalized\u5206\u8d8a\u9ad8\uff0c\u989c\u8272\u8d8a\u6df1\uff0c\u8868\u793a\u8d8a\u9700\u8981\u5173\u6ce8\u3002",
                counts,
            )
        if negative_count > 0:
            inhibition = -values
            return (
                self._select_signed_scores(inhibition, negative_mask),
                "all_negative_abs_raw",
                "\u4e0d\u5b58\u5728\u6b63\u6570patch\u65f6\uff0c\u5bf9attention*gradient\u8d1f\u503c\u53d6\u7edd\u5bf9\u503c\u540e\u6807\u51c6\u5316\uff1bnormalized\u5206\u8d8a\u4f4e\uff0c\u989c\u8272\u8d8a\u6df1\uff0c\u8868\u793a\u8d8a\u9700\u8981\u5173\u6ce8\u3002",
                counts,
            )
        if values.numel() == 0:
            return values.clone(), "empty", "\u6ca1\u6709\u53ef\u7ed8\u5236\u7684patch\u3002", counts
        return (
            torch.zeros_like(values),
            "neutral_zero",
            "\u6b63\u8d1f\u8d21\u732e\u5747\u4e3a0\uff0c\u663e\u793a\u96f6\u6743\u91cd\u3002",
            counts,
        )

    def _rank_heatmap_rgb(self, rank_fraction: float) -> np.ndarray:
        rank_fraction = float(np.clip(rank_fraction, 0.0, 1.0))
        if rank_fraction < 0.25:
            return np.array([205, 230, 255], dtype=np.float32)
        if rank_fraction < 0.50:
            return np.array([255, 244, 170], dtype=np.float32)
        if rank_fraction < 0.75:
            return np.array([255, 165, 48], dtype=np.float32)
        orange_red = np.array([245, 90, 30], dtype=np.float32)
        red = np.array([220, 30, 20], dtype=np.float32)
        t = (rank_fraction - 0.75) / 0.25
        return orange_red * (1.0 - t) + red * t

    def _write_png_rgb(self, path: Path, image: np.ndarray) -> None:
        import struct
        import zlib

        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB image [H, W, 3], got {image.shape}")

        height, width = image.shape[:2]
        raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=6))
            + chunk(b"IEND", b"")
        )
        with open(path, "wb") as handle:
            handle.write(png)

    def _write_patch_csv(
        self,
        path: Path,
        patch_count: int,
        coords: np.ndarray | None,
        columns: dict[str, np.ndarray],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        coords_array = None
        if coords is not None:
            coords_array = np.asarray(coords)
            if coords_array.ndim != 2 or coords_array.shape[0] < patch_count or coords_array.shape[1] < 2:
                coords_array = None

        header = ["patch_index"]
        if coords_array is not None:
            header.extend(["x", "y"])
        header.extend(columns.keys())

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for patch_idx in range(patch_count):
                row = [patch_idx]
                if coords_array is not None:
                    row.extend([float(coords_array[patch_idx, 0]), float(coords_array[patch_idx, 1])])
                for values in columns.values():
                    array = np.asarray(values)
                    row.append(float(array[patch_idx]) if patch_idx < array.shape[0] else "")
                writer.writerow(row)

    #画热力图
    def _render_task_heatmaps(
        self,
        task_scores: dict[str, np.ndarray],
        rank_directions: dict[str, str],
        draw_masks: dict[str, np.ndarray],
        coords: np.ndarray,
        output_dir: Path,
        slide_id: str,
        patch_size: int | None,
    ) -> dict[str, str]:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            print(f"  [TransMIL attention] Invalid coords shape, skipped heatmap rendering: {coords.shape}")
            return {}

        coords = coords[:, :2]
        first_scores = next(iter(task_scores.values()), np.empty((0,), dtype=np.float32))
        if coords.shape[0] != len(first_scores):
            print(
                "  [TransMIL attention] Coord count does not match attention scores, "
                f"skipped heatmaps: coords={coords.shape[0]}, scores={len(first_scores)}"
            )
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        patch_size = int(patch_size or 256)
        min_xy = coords.min(axis=0)
        coords = coords - min_xy.reshape(1, 2)
        extent = coords.max(axis=0) + patch_size
        max_dimension = 1800
        scale = min(max_dimension / max(float(extent[0]), 1.0), max_dimension / max(float(extent[1]), 1.0), 1.0)
        canvas_w = max(1, int(np.ceil(float(extent[0]) * scale)))
        canvas_h = max(1, int(np.ceil(float(extent[1]) * scale)))
        patch_px = max(1, int(np.ceil(patch_size * scale)))
        safe_slide_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(slide_id or "slide"))

        heatmap_paths: dict[str, str] = {}
        for task_name, scores in task_scores.items():
            scores = np.asarray(scores, dtype=np.float32)
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
            draw_mask = np.asarray(draw_masks.get(task_name, scores > 0), dtype=bool)
            draw_indices = np.flatnonzero(draw_mask)
            rank_direction = rank_directions.get(task_name, "high")
            if draw_indices.size and rank_direction == "low":
                order = draw_indices[np.argsort(-scores[draw_indices])]
            else:
                order = draw_indices[np.argsort(scores[draw_indices])] if draw_indices.size else draw_indices
            rank_fractions = np.zeros_like(scores, dtype=np.float32)
            if order.size == 1:
                rank_fractions[order] = 1.0
            elif order.size:
                rank_fractions[order] = np.linspace(0.0, 1.0, int(order.size), dtype=np.float32)
            for patch_idx in order:
                x = int(round(float(coords[patch_idx, 0]) * scale))
                y = int(round(float(coords[patch_idx, 1]) * scale))
                x2 = min(canvas_w - 1, x + patch_px)
                y2 = min(canvas_h - 1, y + patch_px)
                if x2 < x or y2 < y:
                    continue
                rank_fraction = float(rank_fractions[patch_idx])
                color = self._rank_heatmap_rgb(rank_fraction)
                alpha = 0.35 + 0.50 * rank_fraction
                roi = canvas[y:y2 + 1, x:x2 + 1]
                if roi.size == 0:
                    continue
                blended = (roi.astype(np.float32) * (1.0 - alpha) + color.reshape(1, 1, 3) * alpha).astype(np.uint8)
                canvas[y:y2 + 1, x:x2 + 1] = blended

            heatmap_path = output_dir / f"{safe_slide_id}_{task_name.lower()}_grad_attention_heatmap.png"
            self._write_png_rgb(heatmap_path, canvas)
            heatmap_paths[task_name] = str(heatmap_path)

        return heatmap_paths

    #计算attention*gradient，得到热力图结果
    def _build_gradient_attention_result(
        self,
        results: dict,
        pathomics_coords: np.ndarray | None,
        pathomics_patch_size: int | None,
        attention_output_dir: str | None,
        slide_id: str | None,
    ) -> dict:
        attn = results.get('attn2_cls')
        if attn is None:
            raise RuntimeError("TransMIL attention output is missing")

        original_count = int(results['orig_n_patches'])
        square_count = int(results['square_n_patches'])
        token_count = square_count + 1
        internal_padding = int(results.get('attn2_internal_padding', 0) or 0)
        token_start = internal_padding
        token_end = token_start + token_count

        if attn.shape[-1] < token_end:
            raise RuntimeError(
                f"Invalid TransMIL attention shape {tuple(attn.shape)} for token_count={token_count}, "
                f"internal_padding={internal_padding}"
            )

        attn_tokens = attn[..., token_start:token_end]
        attn_patches = attn_tokens[..., 1:]
        shared = self._fold_square_padding(
            attn_patches.detach().mean(dim=1).squeeze(0).squeeze(0),
            original_count,
            square_count,
        )
        shared_np = shared.cpu().numpy().astype(np.float32)

        task_specs = {
            "T": ("t_logits", int(results["t_hat"].item())),
            "N": ("n_logits", int(results["n_hat"].item())),
            "M": ("m_logits", int(results["m_hat"].item())),
            "Stage": ("stage_logits", int(results["stage_hat"].item())),
        }
        task_scores: dict[str, np.ndarray] = {}
        task_raw_scores: dict[str, np.ndarray] = {}
        task_normalized_scores: dict[str, np.ndarray] = {}
        task_heatmap_draw_masks: dict[str, np.ndarray] = {}
        task_gradients: dict[str, np.ndarray] = {}
        score_modes: dict[str, str] = {}
        score_descriptions: dict[str, str] = {}
        score_counts: dict[str, dict[str, int]] = {}
        heatmap_rank_directions: dict[str, str] = {}
        for task_name, (logit_key, class_idx) in task_specs.items():
            target = results[logit_key][0, class_idx]
            grad = torch.autograd.grad(target, attn, retain_graph=True, allow_unused=False)[0]
            grad_patches = grad[..., token_start:token_end][..., 1:]
            folded_grad = self._fold_square_padding(
                grad_patches.detach().mean(dim=1).squeeze(0).squeeze(0),
                original_count,
                square_count,
            )
            weighted_raw = (attn_patches.detach() * grad_patches).mean(dim=1).squeeze(0).squeeze(0)
            folded_raw = self._fold_square_padding(weighted_raw, original_count, square_count)
            folded, score_mode, score_description, counts = self._signed_gradient_scores(folded_raw)
            score_modes[task_name] = score_mode
            score_descriptions[task_name] = score_description
            score_counts[task_name] = counts
            heatmap_rank_directions[task_name] = "low" if score_mode == "all_negative_abs_raw" else "high"
            task_gradients[task_name] = folded_grad.detach().cpu().numpy().astype(np.float32)
            task_raw_scores[task_name] = folded_raw.detach().cpu().numpy().astype(np.float32)
            task_scores[task_name] = folded.detach().cpu().numpy().astype(np.float32)
            task_heatmap_draw_masks[task_name] = task_scores[task_name] > 0

        for task_name, final_scores in task_scores.items():
            if final_scores.size and final_scores.max() > final_scores.min():
                task_normalized_scores[task_name] = (
                    (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
                ).astype(np.float32)
            else:
                task_normalized_scores[task_name] = np.zeros_like(final_scores, dtype=np.float32)

        output_dir = Path(attention_output_dir) if attention_output_dir else (
            Path(config.OUTPUT_DIR) / self.cancer_type / "transmil_attention" / str(slide_id or "unknown_slide")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        coords_array = None
        heatmap_paths: dict[str, str] = {}
        if pathomics_coords is not None:
            coords_array = np.asarray(pathomics_coords)
            if coords_array.ndim == 2 and coords_array.shape[0] >= original_count:
                coords_array = coords_array[:original_count, :2]
                heatmap_paths = self._render_task_heatmaps(
                    task_scores=task_normalized_scores,
                    rank_directions=heatmap_rank_directions,
                    draw_masks=task_heatmap_draw_masks,
                    coords=coords_array,
                    output_dir=output_dir,
                    slide_id=slide_id or "slide",
                    patch_size=pathomics_patch_size,
                )
            else:
                print(
                    "  [TransMIL attention] Coords are missing or shorter than patch tokens, "
                    f"skipped heatmaps: coords_shape={getattr(coords_array, 'shape', None)}, "
                    f"n_patches={original_count}"
                )

        cls_attention_csv = output_dir / "cls_token_attention.csv"
        logits_gradients_csv = output_dir / "logits_gradients.csv"
        attention_gradient_csv = output_dir / "attention_gradient.csv"
        self._write_patch_csv(
            cls_attention_csv,
            original_count,
            coords_array,
            {"cls_attention": shared_np},
        )
        self._write_patch_csv(
            logits_gradients_csv,
            original_count,
            coords_array,
            {
                "t_logit_gradient": task_gradients["T"],
                "n_logit_gradient": task_gradients["N"],
                "m_logit_gradient": task_gradients["M"],
                "stage_logit_gradient": task_gradients["Stage"],
            },
        )
        self._write_patch_csv(
            attention_gradient_csv,
            original_count,
            coords_array,
            {
                "t_raw_attention_gradient": task_raw_scores["T"],
                "t_final_attention_gradient": task_scores["T"],
                "t_normalized_attention_gradient": task_normalized_scores["T"],
                "n_raw_attention_gradient": task_raw_scores["N"],
                "n_final_attention_gradient": task_scores["N"],
                "n_normalized_attention_gradient": task_normalized_scores["N"],
                "m_raw_attention_gradient": task_raw_scores["M"],
                "m_final_attention_gradient": task_scores["M"],
                "m_normalized_attention_gradient": task_normalized_scores["M"],
                "stage_raw_attention_gradient": task_raw_scores["Stage"],
                "stage_final_attention_gradient": task_scores["Stage"],
                "stage_normalized_attention_gradient": task_normalized_scores["Stage"],
            },
        )
        csv_files = {
            "cls_token_attention_csv": str(cls_attention_csv),
            "logits_gradients_csv": str(logits_gradients_csv),
            "attention_gradient_csv": str(attention_gradient_csv),
        }

        attention_npz = output_dir / "gradient_attention.npz"
        np.savez_compressed(
            attention_npz,
            shared_cls_attention=shared_np,
            t_logit_gradient=task_gradients["T"],
            n_logit_gradient=task_gradients["N"],
            m_logit_gradient=task_gradients["M"],
            stage_logit_gradient=task_gradients["Stage"],
            t_grad_attention=task_scores["T"],
            n_grad_attention=task_scores["N"],
            m_grad_attention=task_scores["M"],
            stage_grad_attention=task_scores["Stage"],
            t_normalized_grad_attention=task_normalized_scores["T"],
            n_normalized_grad_attention=task_normalized_scores["N"],
            m_normalized_grad_attention=task_normalized_scores["M"],
            stage_normalized_grad_attention=task_normalized_scores["Stage"],
            t_raw_grad_attention=task_raw_scores["T"],
            n_raw_grad_attention=task_raw_scores["N"],
            m_raw_grad_attention=task_raw_scores["M"],
            stage_raw_grad_attention=task_raw_scores["Stage"],
            coords=coords_array if coords_array is not None else np.empty((0, 2), dtype=np.float32),
        )

        summary_path = output_dir / "attention_summary.json"
        summary_payload = {
            "slide_id": slide_id,
            "method": "Gradient x CLS attention from TransMIL layer2; if any positive patch exists, negative raw scores are ReLUed to zero and higher normalized scores are more important; otherwise negative raw scores are converted to absolute values and lower normalized scores are more important. Heatmap colors are assigned by within-task patch rank quartiles from light blue to yellow, orange, and red without patch borders.",
            "patch_count": original_count,
            "square_patch_count": square_count,
            "square_pad_length": int(results.get("square_pad_length", 0) or 0),
            "nystrom_internal_padding": internal_padding,
            "attention_npz": str(attention_npz),
            "csv_files": csv_files,
            "heatmaps": heatmap_paths,
            "score_modes": score_modes,
            "score_descriptions": score_descriptions,
            "score_counts": score_counts,
            "heatmap_rank_directions": heatmap_rank_directions,
            "tasks": {
                task_name: {
                    "score_mode": score_modes.get(task_name, "unknown"),
                    "score_description": score_descriptions.get(task_name, ""),
                    "score_counts": score_counts.get(task_name, {}),
                    "heatmap_rank_direction": heatmap_rank_directions.get(task_name, "high"),
                    "raw_signed_min": float(task_raw_scores[task_name].min()) if task_raw_scores[task_name].size else 0.0,
                    "raw_signed_max": float(task_raw_scores[task_name].max()) if task_raw_scores[task_name].size else 0.0,
                    "raw_signed_mean": float(task_raw_scores[task_name].mean()) if task_raw_scores[task_name].size else 0.0,
                    "raw_min": float(scores.min()) if scores.size else 0.0,
                    "raw_max": float(scores.max()) if scores.size else 0.0,
                    "raw_mean": float(scores.mean()) if scores.size else 0.0,
                    "normalized_min": float(task_normalized_scores[task_name].min()) if task_normalized_scores[task_name].size else 0.0,
                    "normalized_max": float(task_normalized_scores[task_name].max()) if task_normalized_scores[task_name].size else 0.0,
                    "normalized_mean": float(task_normalized_scores[task_name].mean()) if task_normalized_scores[task_name].size else 0.0,
                }
                for task_name, scores in task_scores.items()
            },
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

        return {
            **summary_payload,
            "attention_summary_json": str(summary_path),
            "shared_cls_attention": shared_np.tolist(),
            "task_logit_gradients": {name: gradients.tolist() for name, gradients in task_gradients.items()},
            "task_attention": {name: scores.tolist() for name, scores in task_scores.items()},
            "task_normalized_attention": {name: scores.tolist() for name, scores in task_normalized_scores.items()},
            "heatmap_rank_directions": heatmap_rank_directions,
        }

    def predict(
        self,
        features: np.ndarray,
        return_attention: bool = False,
        pathomics_coords: np.ndarray | None = None,
        pathomics_patch_size: int | None = None,
        attention_output_dir: str | None = None,
        slide_id: str | None = None,
    ) -> dict:
        if self.model is None:
            self.load_model()

        feature_tensor = self._coerce_features(features).to(self.device)
        if not return_attention:
            with torch.no_grad():
                results = self.model(data=feature_tensor.unsqueeze(0))
                return self._format_predictions(results)

        self.model.zero_grad(set_to_none=True)
        results = self.model(data=feature_tensor.unsqueeze(0), return_attn=True)
        predictions = self._format_predictions(results)
        predictions["attention"] = self._build_gradient_attention_result(
            results=results,
            pathomics_coords=pathomics_coords,
            pathomics_patch_size=pathomics_patch_size,
            attention_output_dir=attention_output_dir,
            slide_id=slide_id,
        )
        self.model.zero_grad(set_to_none=True)
        return predictions

# 全局预测器实例
_predictors = {}

def get_transmil_predictor(cancer_type: str = DEFAULT_CANCER_TYPE) -> TransMILPredictor:
    """获取全局预测器实例"""
    if cancer_type not in _predictors:
        _predictors[cancer_type] = TransMILPredictor(cancer_type=cancer_type)
    return _predictors[cancer_type]

def predict_tnm_stage(features, cancer_type: str) -> str:
    """
    TransMIL预测工具函数 - 供LangChain Tool调用
    
    Args:
        features: 病理特征，格式为numpy数组或特征文件路径
        
    Returns:
        str: 格式化的预测结果
    """
    predictor = get_predictor(cancer_type)
    
    # 支持文件路径
    if isinstance(features, str) and os.path.exists(features):
        result = predictor.predict_from_features_file(features)
    else:
        result = predictor.predict_raw_features(features)
    
    # 格式化输出
    output_lines = [
        "="*40,
        "TransMIL 病理分析结果",
        "="*40,
        f"T分期: {result['T']['label']} (置信度: {result['T']['confidence']:.2%})",
        f"N分期: {result['N']['label']} (置信度: {result['N']['confidence']:.2%})",
        f"M分期: {result['M']['label']} (置信度: {result['M']['confidence']:.2%})",
        f"Stage: {result['Stage']['label']} (置信度: {result['Stage']['confidence']:.2%})",
        "="*40,
        "各分期概率分布:",
        f"  T: {dict(zip(config.T_LABELS, [f'{p:.1%}' for p in result['T']['prob']]))}",
        f"  N: {dict(zip(config.N_LABELS, [f'{p:.1%}' for p in result['N']['prob']]))}",
        f"  M: {dict(zip(config.M_LABELS, [f'{p:.1%}' for p in result['M']['prob']]))}",
        f"  Stage: {dict(zip(config.STAGE_LABELS, [f'{p:.1%}' for p in result['Stage']['prob']]))}",
    ]
    
    return "\n".join(output_lines)


if __name__ == "__main__":
    # 测试
    # 创建随机特征测试
    test_features = np.random.randn(500, 1024)
    result = predict_tnm_stage(test_features)
    print(result)
