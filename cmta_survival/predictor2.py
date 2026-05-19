import os
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import config
from cmta_survival.network import CMTA


SUPPORTED_CANCER_TYPES = list(getattr(config, "SUPPORTED_CANCER_TYPES", config.MODEL_CONFIGS.keys()))
DEFAULT_CANCER_TYPE = "BRCA"
DEVICE = "cpu"

SIGNATURE_GROUPS = [
    "Tumor Suppressor Genes",
    "Oncogenes",
    "Protein Kinases",
    "Cell Differentiation Markers",
    "Transcription Factors",
    "Cytokines and Growth Factors",
]

NON_GENOMIC_COLUMNS = {
    "case_id",
    "slide_id",
    "site",
    "s_female",
    "is_female",
    "oncotree_code",
    "age",
    "survival_months",
    "censorship",
    "train",
}


class CMTAInputError(ValueError):
    """Raised when CMTA inputs or resources are missing."""


def _preview_items(items: List[str], max_items: int = 20) -> str:
    if not items:
        return "none"
    preview = ", ".join(str(item) for item in items[:max_items])
    if len(items) > max_items:
        preview += f", ... total {len(items)}"
    return preview


def _raise_cmta_missing_info(context: str, missing_items: List[str]) -> None:
    lines = [f"[CMTA input error] {context}", "Missing or invalid items:"]
    lines.extend(f"- {item}" for item in missing_items)
    message = "\n".join(lines)
    print(message)
    raise CMTAInputError(message)


class CMSurvivalPredictor:
    """CMTA survival predictor.

    Web/API mode only accepts the user's uploaded single-patient genomic CSV
    or six already prepared omic arrays. Built-in patient lookup fallback is
    intentionally disabled for open-source deployment.
    """

    def __init__(
        self,
        cancer_type: str = DEFAULT_CANCER_TYPE,
        model_size: str = "small",
        fusion: str = "concat",
        device: str = DEVICE,
    ):
        self.cancer_type = cancer_type.upper()
        self.model_size = model_size
        self.fusion = fusion
        self.device = device

        if self.cancer_type not in SUPPORTED_CANCER_TYPES:
            raise ValueError(f"Unsupported cancer type: {self.cancer_type}. Supported: {SUPPORTED_CANCER_TYPES}")

        self.model = None
        self.model_path = None
        self.preprocessing: dict[str, Any] = {}
        self.q_bins: list[float] | None = None
        self.omic_sizes = [0, 0, 0, 0, 0, 0]
        self._load_model()

    def _load_model(self) -> None:
        try:
            from .model_loader import find_best_model_path, load_cmta_checkpoint

            model_path = find_best_model_path(self.cancer_type)
            if model_path is None or not os.path.exists(model_path):
                expected_dir = config.MODEL_CONFIGS[self.cancer_type]["result_dir"]
                _raise_cmta_missing_info(
                    "CMTA checkpoint was not found",
                    [f"cancer_type={self.cancer_type}", f"expected_dir={expected_dir}"],
                )

            checkpoint, inferred_omic_sizes, preprocessing = load_cmta_checkpoint(model_path, self.device)
            self.model_path = model_path
            self.preprocessing = preprocessing or {}
            self.omic_sizes = list(self.preprocessing.get("omic_sizes") or inferred_omic_sizes)
            self.q_bins = self.preprocessing.get("q_bins")
            state_dict = checkpoint.get("state_dict", checkpoint)

            self.model = CMTA(
                omic_sizes=self.omic_sizes,
                n_classes=4,
                fusion=self.fusion,
                model_size=self.model_size,
            )
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

        except CMTAInputError:
            raise
        except Exception as exc:
            _raise_cmta_missing_info("CMTA model loading failed", [f"{type(exc).__name__}: {exc}"])

    def predict_survival(
        self,
        pathomics_features: np.ndarray,
        case_id: str | None = None,
        slide_id: str | None = None,
        genomic_features: list | str | None = None,
        return_probs: bool = True,
        return_attention: bool = False,
        pathomics_coords: np.ndarray | None = None,
        pathomics_patch_size: int | None = None,
        attention_output_dir: str | None = None,
    ) -> Dict[str, Any]:
        if pathomics_features is None:
            _raise_cmta_missing_info("Missing pathomics input", ["pathomics_features is empty"])

        if hasattr(pathomics_features, "cpu"):
            pathomics_features = pathomics_features.cpu().numpy()
        pathomics_features = np.asarray(pathomics_features, dtype=np.float32)
        if pathomics_features.ndim == 3 and pathomics_features.shape[0] == 1:
            pathomics_features = pathomics_features[0]
        if pathomics_features.ndim != 2 or pathomics_features.shape[1] != 1024:
            _raise_cmta_missing_info(
                "Invalid pathomics feature shape",
                [f"expected [n_patches, 1024], got {list(pathomics_features.shape)}"],
            )

        if isinstance(genomic_features, str):
            if not genomic_features.lower().endswith(".csv"):
                _raise_cmta_missing_info(
                    "Invalid genomic input",
                    [f"expected a single-patient genomic CSV path, got {genomic_features}"],
                )
            prepared_genomic_features = self._load_genomic_from_csv(genomic_features)
        elif isinstance(genomic_features, (list, tuple)):
            prepared_genomic_features = list(genomic_features)
        else:
            _raise_cmta_missing_info(
                "Missing genomic input",
                [
                    "Upload a single-patient genomic CSV or provide six omic feature arrays.",
                    "Built-in CSV lookup by case_id or slide_id is not supported.",
                ],
            )

        self._validate_genomic_features(prepared_genomic_features)

        self.model.eval()
        with torch.no_grad():
            x_path = torch.tensor(pathomics_features, dtype=torch.float32).to(self.device)
            x_omic = [
                torch.tensor(feature, dtype=torch.float32).to(self.device).view(-1)
                for feature in prepared_genomic_features
            ]

            print(f"[CMTA输入数据] path shape: {x_path.shape}, omic shape: ({x_omic[0].shape},{x_omic[1].shape},{x_omic[2].shape},{x_omic[3].shape},{x_omic[4].shape},{x_omic[5].shape})")

            model_output = self.model(
                x_path=x_path,
                **{f"x_omic{i + 1}": x_omic[i] for i in range(6)},
                return_attn=return_attention,
            )
            if return_attention:
                hazards, survival_probs, _, _, _, _, attention = model_output
            else:
                hazards, survival_probs, _, _, _, _ = model_output
                attention = None

            survival_np = survival_probs.cpu().numpy()[0]
            survival_months = self._calculate_survival_time(survival_np)
            risk_score = float(-torch.sum(survival_probs, dim=1).cpu().numpy()[0])
            risk_class = self._estimate_risk_class(survival_np, survival_months)

            result = {
                "cancer_type": self.cancer_type,
                "case_id": case_id,
                "slide_id": slide_id,
                "predicted_survival_months": float(survival_months),
                "risk_level": self._get_risk_label(risk_class),
                "risk_class": risk_class,
                "risk_score": risk_score,
                "hazard_probs": hazards.cpu().numpy()[0].tolist(),
            }
            if return_probs:
                result["survival_probs"] = survival_np.tolist()
            if self.q_bins is not None:
                result["q_bins"] = [float(value) for value in self.q_bins]
            if return_attention and attention is not None:
                result["attention"] = self._build_attention_result(
                    attention=attention,
                    n_path_tokens=pathomics_features.shape[0],
                    pathomics_coords=pathomics_coords,
                    pathomics_patch_size=pathomics_patch_size,
                    attention_output_dir=attention_output_dir,
                    slide_id=slide_id,
                )
            return result

    def _validate_genomic_features(self, genomic_features: list) -> None:
        missing_items: List[str] = []

        if genomic_features is None:
            _raise_cmta_missing_info("Genomic features were not loaded", ["genomic_features is empty"])

        if len(self.omic_sizes) != 6:
            missing_items.append(f"model omic_sizes must contain 6 values, got {len(self.omic_sizes)}: {self.omic_sizes}")

        if len(genomic_features) != 6:
            missing_items.append(f"expected 6 omic inputs, got {len(genomic_features)}")

        for i, expected_size in enumerate(self.omic_sizes[:6]):
            group_name = SIGNATURE_GROUPS[i]
            if i >= len(genomic_features):
                missing_items.append(f"omic{i + 1}({group_name}) is missing")
                continue

            feature = genomic_features[i]
            if feature is None:
                missing_items.append(f"omic{i + 1}({group_name}) is empty")
                continue

            arr = np.asarray(feature)
            if arr.size == 0:
                missing_items.append(f"omic{i + 1}({group_name}) has no values")
                continue

            try:
                numeric_arr = arr.astype(np.float32, copy=False)
            except (TypeError, ValueError):
                missing_items.append(f"omic{i + 1}({group_name}) contains non-numeric values")
                continue

            if np.isnan(numeric_arr).any():
                missing_items.append(f"omic{i + 1}({group_name}) contains NaN values")
                continue

            actual_size = numeric_arr.reshape(-1).shape[0]
            if actual_size != expected_size:
                missing_items.append(
                    f"omic{i + 1}({group_name}) size mismatch: expected {expected_size}, got {actual_size}"
                )

        if missing_items:
            _raise_cmta_missing_info("Genomic features do not match CMTA input requirements", missing_items)

    def _has_checkpoint_preprocessing(self) -> bool:
        return bool(
            self.preprocessing
            and self.preprocessing.get("genomic_feature_columns")
            and self.preprocessing.get("omic_names")
            and self.preprocessing.get("scaler")
        )

    def _load_genomic_from_checkpoint_metadata(self, df: pd.DataFrame) -> list:
        feature_columns = list(self.preprocessing.get("genomic_feature_columns") or [])
        omic_names = [list(cols) for cols in self.preprocessing.get("omic_names") or []]
        scaler = self.preprocessing.get("scaler") or {}

        missing_items: List[str] = []
        if len(omic_names) != 6:
            missing_items.append(f"preprocessing omic_names must contain 6 groups, got {len(omic_names)}")

        if not feature_columns:
            missing_items.append("preprocessing genomic_feature_columns is empty")

        mean = np.asarray(scaler.get("mean", []), dtype=np.float32)
        scale = np.asarray(scaler.get("scale", []), dtype=np.float32)
        if mean.shape[0] != len(feature_columns):
            missing_items.append(f"scaler mean length mismatch: expected {len(feature_columns)}, got {mean.shape[0]}")
        if scale.shape[0] != len(feature_columns):
            missing_items.append(f"scaler scale length mismatch: expected {len(feature_columns)}, got {scale.shape[0]}")

        feature_index = {col: idx for idx, col in enumerate(feature_columns)}
        required_omic_cols: List[str] = []
        seen_required_cols: set[str] = set()
        for omic_cols in omic_names:
            for col in omic_cols:
                if col not in seen_required_cols:
                    seen_required_cols.add(col)
                    required_omic_cols.append(col)

        missing_metadata_cols = [col for col in required_omic_cols if col not in feature_index]
        if missing_metadata_cols:
            missing_items.append(
                "checkpoint omic_names contain columns not found in scaler feature_columns: "
                f"{_preview_items(missing_metadata_cols)}"
            )

        source_columns: dict[str, str] = {}
        missing_required_cols: List[str] = []
        for col in required_omic_cols:
            if col in df.columns:
                source_columns[col] = col
                continue
            if col.endswith("_mut") and col[:-4] in df.columns:
                source_columns[col] = col[:-4]
                continue
            missing_required_cols.append(col)
        if missing_required_cols:
            missing_items.append(
                "uploaded genomic CSV is missing required CMTA omic columns: "
                f"{_preview_items(missing_required_cols)}"
            )

        if missing_items:
            _raise_cmta_missing_info("Checkpoint preprocessing metadata cannot be applied", missing_items)

        feature_df = pd.DataFrame(
            {
                col: pd.to_numeric(df[source_columns[col]], errors="coerce")
                for col in required_omic_cols
            }
        )
        invalid_cols = feature_df.columns[feature_df.isna().any()].tolist()
        if invalid_cols:
            _raise_cmta_missing_info(
                "Uploaded genomic CSV contains missing or non-numeric required CMTA omic features",
                [f"invalid columns: {_preview_items(invalid_cols)}"],
            )

        scaled_columns = {}
        for col in required_omic_cols:
            idx = feature_index[col]
            safe_scale = scale[idx] if scale[idx] != 0 else 1.0
            values = feature_df[col].to_numpy(dtype=np.float32)
            scaled_columns[col] = (values - mean[idx]) / safe_scale
        scaled_df = pd.DataFrame(scaled_columns, columns=required_omic_cols)

        genomic_features = []
        for i, omic_cols in enumerate(omic_names):
            expected_size = self.omic_sizes[i] if i < len(self.omic_sizes) else len(omic_cols)
            missing_omic_cols = [col for col in omic_cols if col not in scaled_df.columns]
            if missing_omic_cols:
                _raise_cmta_missing_info(
                    "Checkpoint omic feature columns are not present after scaling",
                    [f"omic{i + 1}: {_preview_items(missing_omic_cols)}"],
                )
            if len(omic_cols) != expected_size:
                _raise_cmta_missing_info(
                    "Checkpoint omic_names do not match model input dimensions",
                    [f"omic{i + 1}: model expects {expected_size}, metadata has {len(omic_cols)}"],
                )
            genomic_features.append(scaled_df[omic_cols].to_numpy(dtype=np.float32).reshape(1, -1))

        print("[Genomics数据对齐] Genomic CSV parsed with checkpoint preprocessing metadata")
        return genomic_features

    def _load_genomic_from_csv(self, csv_path: str) -> list:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            _raise_cmta_missing_info("Cannot load genomic CSV", [f"CSV file does not exist: {csv_path}"])

        df = pd.read_csv(csv_file)
        if df.empty:
            _raise_cmta_missing_info("Genomic CSV is empty", [f"CSV file: {csv_path}"])
        if len(df) != 1:
            _raise_cmta_missing_info(
                "Genomic CSV must contain exactly one patient row",
                [f"current rows: {len(df)}", "Upload a single-patient genomic CSV."],
            )

        if self._has_checkpoint_preprocessing():
            return self._load_genomic_from_checkpoint_metadata(df)

        signature_path = Path(config.SIGNATURES_CSV_PATH) if config.SIGNATURES_CSV_PATH else None
        if signature_path is None or not signature_path.exists():
            _raise_cmta_missing_info(
                "Cannot load signature grouping file",
                [f"signatures.csv not found: {signature_path or 'not configured'}"],
            )

        signatures = pd.read_csv(signature_path)
        all_columns = df.columns.tolist()
        column_set = set(all_columns)
        cnv_columns = {col for col in all_columns if col.endswith("_cnv")}
        rnaseq_columns = {col for col in all_columns if col.endswith("_rnaseq")}
        snv_columns = {
            col
            for col in all_columns
            if not col.endswith("_cnv") and not col.endswith("_rnaseq") and col not in NON_GENOMIC_COLUMNS
        }

        print(
            f"[Genomics数据加载] Uploaded genomic CSV: {len(cnv_columns)} CNV, "
            f"{len(snv_columns)} SNV, {len(rnaseq_columns)} RNAseq"
        )

        genomic_features = []
        missing_items: List[str] = []
        if len(self.omic_sizes) != 6:
            _raise_cmta_missing_info(
                "Model omic_sizes are incomplete",
                [f"expected 6 values, got {len(self.omic_sizes)}: {self.omic_sizes}"],
            )

        for i, group_name in enumerate(SIGNATURE_GROUPS):
            expected_size = self.omic_sizes[i]

            if group_name not in signatures.columns:
                missing_items.append(f"signatures.csv missing group column: {group_name}")
                genomic_features.append(None)
                continue

            genes = [str(gene).strip() for gene in signatures[group_name].dropna().tolist() if str(gene).strip()]
            if not genes:
                missing_items.append(f"signatures.csv group has no genes: {group_name}")
                genomic_features.append(None)
                continue

            group_cols: List[str] = []
            expected_cols: List[str] = []
            for gene in genes:
                expected_cols.extend([gene, f"{gene}_cnv", f"{gene}_rnaseq"])
                if gene in snv_columns:
                    group_cols.append(gene)
                cnv_col = f"{gene}_cnv"
                if cnv_col in cnv_columns:
                    group_cols.append(cnv_col)
                rnaseq_col = f"{gene}_rnaseq"
                if rnaseq_col in rnaseq_columns:
                    group_cols.append(rnaseq_col)

            if len(group_cols) < expected_size:
                missing_cols = [col for col in expected_cols if col not in column_set]
                missing_items.append(
                    f"omic{i + 1}({group_name}) has too few matched features: "
                    f"expected {expected_size}, matched {len(group_cols)}, "
                    f"missing example columns: {_preview_items(missing_cols)}"
                )
                genomic_features.append(None)
                continue

            group_cols = group_cols[:expected_size]
            group_df = df[group_cols].apply(pd.to_numeric, errors="coerce")
            invalid_cols = group_df.columns[group_df.isna().any()].tolist()
            if invalid_cols:
                missing_items.append(
                    f"omic{i + 1}({group_name}) contains missing or non-numeric columns: "
                    f"{_preview_items(invalid_cols)}"
                )
                genomic_features.append(None)
                continue

            genomic_features.append(group_df.to_numpy(dtype=np.float32).reshape(1, -1))

        if missing_items:
            _raise_cmta_missing_info("Uploaded genomic CSV cannot satisfy CMTA input requirements", missing_items)

        return genomic_features

    def _get_omic_group_labels(self) -> list[str]:
        signature_columns = self.preprocessing.get("signature_columns") if self.preprocessing else None
        if isinstance(signature_columns, list) and len(signature_columns) == 6:
            return [str(name) for name in signature_columns]
        return list(SIGNATURE_GROUPS)

    def _attention_to_numpy(self, tensor: Any, name: str) -> np.ndarray:
        if tensor is None:
            _raise_cmta_missing_info("CMTA attention output is missing", [name])
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        array = np.asarray(tensor, dtype=np.float32)
        if array.ndim == 4:
            array = array.mean(axis=1)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            _raise_cmta_missing_info("Invalid CMTA attention shape", [f"{name} shape={list(array.shape)}"])
        return array

    def _summarize_p_in_g_attention(self, p_in_g_att: np.ndarray) -> dict[str, Any]:
        labels = self._get_omic_group_labels()
        if p_in_g_att.shape[1] != len(labels):
            _raise_cmta_missing_info(
                "Path-to-genomic attention does not match omic groups",
                [f"attention columns={p_in_g_att.shape[1]}, omic labels={len(labels)}"],
            )

        top_indices = np.argmax(p_in_g_att, axis=1)
        top_counts = np.bincount(top_indices, minlength=len(labels))
        mean_attention = p_in_g_att.mean(axis=0)
        patch_count = int(p_in_g_att.shape[0])

        omics = []
        for idx in range(len(labels)):
            count = int(top_counts[idx])
            omics.append(
                {
                    "omic_index": idx + 1,
                    "omic_name": labels[idx],
                    "top_patch_count": count,
                    "top_patch_fraction": float(count / patch_count) if patch_count else 0.0,
                    "mean_attention": float(mean_attention[idx]),
                }
            )

        omics.sort(key=lambda item: (-item["top_patch_count"], -item["mean_attention"], item["omic_index"]))
        return {
            "patch_count": patch_count,
            "top_omic": omics[0] if omics else None,
            "omics": omics,
        }

    def _column_to_gene_name(self, column: str) -> str:
        gene = str(column)
        for suffix in ("_rnaseq", "_cnv", "_mut"):
            if gene.endswith(suffix):
                return gene[: -len(suffix)]
        return gene

    def _get_omic_gene_names(self, omic_index: int, max_genes: int = 16) -> list[str]:
        omic_names = self.preprocessing.get("omic_names") if self.preprocessing else None
        if not isinstance(omic_names, list) or omic_index < 1 or omic_index > len(omic_names):
            return []

        columns = omic_names[omic_index - 1]
        if not isinstance(columns, list):
            return []

        genes: list[str] = []
        seen: set[str] = set()
        for column in columns:
            gene = self._column_to_gene_name(str(column)).strip()
            if not gene or gene in seen:
                continue
            seen.add(gene)
            genes.append(gene)
            if len(genes) >= max_genes:
                break
        return genes

    #标准化attention值
    def _normalize_heat_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value - min_value < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return (values - min_value) / (max_value - min_value)

    #将attention值按大小分配比例
    def _rank_heat_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        if values.size == 1:
            return np.ones_like(values, dtype=np.float32)

        series = pd.Series(values)
        ranks = series.rank(method="average", ascending=True).to_numpy(dtype=np.float32)
        return (ranks - 1.0) / float(values.size - 1)

    #将attention值按比例分配颜色
    def _rank_heat_rgb(self, value: float) -> np.ndarray:
        value = float(np.clip(value, 0.0, 1.0))
        anchors = np.array(
            [
                [0.70, 0.86, 1.00],  # lowest 25%: light blue
                [1.00, 0.96, 0.55],  # 25%-50%: light yellow
                [1.00, 0.68, 0.18],  # 50%-75%: yellow-orange
                [0.95, 0.25, 0.08],  # top 25%: orange-red
                [0.82, 0.00, 0.00],  # highest patches: red
            ],
            dtype=np.float32,
        )
        positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
        channel_values = [np.interp(value, positions, anchors[:, channel]) for channel in range(3)]
        return np.array(channel_values, dtype=np.float32) * 255.0

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

    #画热力图
    def _render_omic_patch_heatmaps(
        self,
        g_in_p_att: np.ndarray,
        coords: np.ndarray,
        output_dir: Path,
        slide_id: str,
        patch_size: int | None,
    ) -> list[str]:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            print(f"  [CMTA attention] Invalid coords shape, skipped heatmap rendering: {coords.shape}")
            return []
        coords = coords[:, :2]
        if coords.shape[0] != g_in_p_att.shape[1]:
            print(
                "  [CMTA attention] Coord count does not match g_in_p attention tokens, "
                f"skipped heatmaps: coords={coords.shape[0]}, attn={g_in_p_att.shape[1]}"
            )
            return []

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
        labels = self._get_omic_group_labels()
        safe_slide_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(slide_id or "slide"))

        heatmap_paths: list[str] = []
        for omic_idx, weights in enumerate(g_in_p_att[:6]):
            weights = np.asarray(weights, dtype=np.float32)
            rank_heat_values = self._rank_heat_values(weights)
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
            order = np.argsort(rank_heat_values)
            for patch_idx in order:
                x = int(round(float(coords[patch_idx, 0]) * scale))
                y = int(round(float(coords[patch_idx, 1]) * scale))
                x2 = min(canvas_w - 1, x + patch_px)
                y2 = min(canvas_h - 1, y + patch_px)
                heat_value = float(rank_heat_values[patch_idx])
                color = self._rank_heat_rgb(heat_value)
                alpha = 0.20 + 0.80 * heat_value
                roi = canvas[y:y2 + 1, x:x2 + 1]
                if roi.size == 0:
                    continue
                blended = (roi.astype(np.float32) * (1.0 - alpha) + color.reshape(1, 1, 3) * alpha).astype(np.uint8)
                canvas[y:y2 + 1, x:x2 + 1] = blended

            omic_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in labels[omic_idx])
            heatmap_path = output_dir / f"{safe_slide_id}_omic{omic_idx + 1}_{omic_label}_g_in_p_heatmap.png"
            self._write_png_rgb(heatmap_path, canvas)
            heatmap_paths.append(str(heatmap_path))

        return heatmap_paths

    def _write_g_in_p_attention_csv(
        self,
        g_in_p_att: np.ndarray,
        coords: np.ndarray | None,
        output_dir: Path,
    ) -> Path:
        labels = self._get_omic_group_labels()
        n_patches = int(g_in_p_att.shape[1])
        coords_array = None
        if coords is not None:
            coords_array = np.asarray(coords, dtype=np.float32)
            if coords_array.ndim == 2 and coords_array.shape[0] >= n_patches and coords_array.shape[1] >= 2:
                coords_array = coords_array[:n_patches, :2]
            else:
                coords_array = None

        rows = []
        for omic_idx, weights in enumerate(g_in_p_att[:6]):
            normalized = self._normalize_heat_values(weights)
            rank_heat_values = self._rank_heat_values(weights)
            omic_name = labels[omic_idx] if omic_idx < len(labels) else f"omic{omic_idx + 1}"
            for patch_idx, attention_weight in enumerate(weights[:n_patches]):
                row = {
                    "omic_index": omic_idx + 1,
                    "omic_name": omic_name,
                    "patch_index": patch_idx,
                    "attention_weight": float(attention_weight),
                    "normalized_attention_weight": float(normalized[patch_idx]),
                    "heatmap_rank_percentile": float(rank_heat_values[patch_idx]),
                }
                if coords_array is not None:
                    row["coord_x"] = float(coords_array[patch_idx, 0])
                    row["coord_y"] = float(coords_array[patch_idx, 1])
                rows.append(row)

        csv_path = output_dir / "g_in_p_att_softmax.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def _build_attention_result(
        self,
        attention: dict[str, Any],
        n_path_tokens: int,
        pathomics_coords: np.ndarray | None,
        pathomics_patch_size: int | None,
        attention_output_dir: str | None,
        slide_id: str | None,
    ) -> dict[str, Any]:
        p_in_g_att = self._attention_to_numpy(attention.get("p_in_g_att"), "p_in_g_att")
        g_in_p_att = self._attention_to_numpy(attention.get("g_in_p_att"), "g_in_p_att")

        n_path_tokens = int(n_path_tokens)
        p_in_g_att = p_in_g_att[:n_path_tokens, :6]
        g_in_p_att = g_in_p_att[:6, :n_path_tokens]
        summary = self._summarize_p_in_g_attention(p_in_g_att)
        top_omic = summary.get("top_omic") or {}
        top_omic_index = int(top_omic.get("omic_index", 0) or 0)
        top_omic_genes = self._get_omic_gene_names(top_omic_index)

        output_dir = Path(attention_output_dir) if attention_output_dir else (
            Path(config.OUTPUT_DIR) / self.cancer_type / "cmta_attention" / str(slide_id or "unknown_slide")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        coords_array = None
        heatmap_paths: list[str] = []
        if pathomics_coords is not None:
            coords_array = np.asarray(pathomics_coords)
            if coords_array.ndim == 2 and coords_array.shape[0] >= n_path_tokens:
                coords_array = coords_array[:n_path_tokens, :2]
                heatmap_paths = self._render_omic_patch_heatmaps(
                    g_in_p_att=g_in_p_att,
                    coords=coords_array,
                    output_dir=output_dir,
                    slide_id=slide_id or "slide",
                    patch_size=pathomics_patch_size,
                )
            else:
                print(
                    "  [CMTA attention] Coords are missing or shorter than sampled path tokens, "
                    f"skipped heatmaps: coords_shape={getattr(coords_array, 'shape', None)}, n_path_tokens={n_path_tokens}"
                )

        g_in_p_attention_csv = self._write_g_in_p_attention_csv(g_in_p_att, coords_array, output_dir)
        attention_npz = output_dir / "cross_attention.npz"
        np.savez_compressed(
            attention_npz,
            p_in_g_att=p_in_g_att,
            g_in_p_att=g_in_p_att,
            coords=coords_array if coords_array is not None else np.empty((0, 2), dtype=np.float32),
        )

        summary_path = output_dir / "attention_summary.json"
        summary_payload = {
            "slide_id": slide_id,
            "omic_labels": self._get_omic_group_labels(),
            "p_in_g_summary": summary,
            "top_omic_genes": {
                "omic_index": top_omic_index,
                "omic_name": top_omic.get("omic_name"),
                "genes": top_omic_genes,
            },
            "g_in_p_heatmaps": heatmap_paths,
            "g_in_p_attention_csv": str(g_in_p_attention_csv),
            "attention_npz": str(attention_npz),
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

        return {
            **summary_payload,
            "attention_summary_json": str(summary_path),
            "p_in_g_att": p_in_g_att.tolist(),
            "g_in_p_att": g_in_p_att.tolist(),
        }

    def _calculate_survival_time(self, survival_probs: np.ndarray) -> float:
        if self.q_bins is not None and len(self.q_bins) == len(survival_probs) + 1:
            q_bins = np.asarray(self.q_bins, dtype=np.float32)
            prev_t = float(q_bins[0])
            prev_s = 1.0
            for i, cur_s in enumerate(survival_probs):
                cur_t = float(q_bins[i + 1])
                cur_s = float(cur_s)
                if cur_s <= 0.5:
                    denom = cur_s - prev_s
                    if abs(denom) < 1e-8:
                        return cur_t
                    fraction = (0.5 - prev_s) / denom
                    fraction = float(np.clip(fraction, 0.0, 1.0))
                    return float(prev_t + (cur_t - prev_t) * fraction)
                prev_t = cur_t
                prev_s = cur_s
            return float(q_bins[-1])

        time_points = np.array([12, 24, 36, 48], dtype=np.float32)

        if survival_probs[-1] > 0.5:
            return float(60 * (1 + survival_probs[-1]))

        idx = np.where(survival_probs <= 0.5)[0]
        if len(idx) == 0:
            return 60.0

        i = int(idx[0])
        if i == 0:
            return float(time_points[0] * 0.5)

        prev_s = survival_probs[i - 1]
        cur_s = survival_probs[i]
        prev_t = time_points[i - 1]
        cur_t = time_points[i]
        return float(prev_t + (cur_t - prev_t) * (0.5 - prev_s) / (cur_s - prev_s + 1e-8))

    def _estimate_risk_class(self, survival_probs: np.ndarray, survival_months: float) -> int:
        if self.q_bins is not None and len(self.q_bins) == len(survival_probs) + 1:
            q_bins = np.asarray(self.q_bins, dtype=np.float32)
            survival_bin = int(np.searchsorted(q_bins[1:-1], survival_months, side="right"))
            survival_bin = int(np.clip(survival_bin, 0, len(survival_probs) - 1))
            return int((len(survival_probs) - 1) - survival_bin)

        mean_survival = float(np.mean(survival_probs))
        if mean_survival >= 0.75:
            return 0
        if mean_survival >= 0.50:
            return 1
        if mean_survival >= 0.25:
            return 2
        return 3

    def _get_risk_label(self, risk_class: int) -> str:
        labels = ["low risk", "low-medium risk", "medium-high risk", "high risk"]
        if risk_class < 0 or risk_class >= len(labels):
            return "unknown risk"
        return labels[risk_class]


_predictors: dict[str, CMSurvivalPredictor] = {}


def get_predictor(cancer_type: str = DEFAULT_CANCER_TYPE) -> CMSurvivalPredictor:
    key = cancer_type.upper()
    if key not in _predictors:
        _predictors[key] = CMSurvivalPredictor(cancer_type=key)
    return _predictors[key]


def predict_survival(
    cancer_type: str,
    pathomics_features: np.ndarray | str,
    case_id: str | None = None,
    slide_id: str | None = None,
    genomic_features: list | str | None = None,
    return_attention: bool = False,
    pathomics_coords: np.ndarray | None = None,
    pathomics_patch_size: int | None = None,
    attention_output_dir: str | None = None,
) -> Dict[str, Any]:
    if isinstance(pathomics_features, str):
        path = Path(pathomics_features)
        if path.suffix == ".npy":
            pathomics_features = np.load(path)
        elif path.suffix == ".pt":
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                loaded = loaded.get("features", loaded.get("data", loaded))
            pathomics_features = loaded.cpu().numpy() if hasattr(loaded, "cpu") else loaded
        else:
            _raise_cmta_missing_info("Unsupported pathomics feature file", [f"path={pathomics_features}"])

    predictor = get_predictor(cancer_type)
    return predictor.predict_survival(
        pathomics_features,
        case_id=case_id,
        slide_id=slide_id,
        genomic_features=genomic_features,
        return_attention=return_attention,
        pathomics_coords=pathomics_coords,
        pathomics_patch_size=pathomics_patch_size,
        attention_output_dir=attention_output_dir,
    )


def format_survival_result(result: Dict[str, Any]) -> str:
    patient_info = f", patient ID: {result.get('case_id')}" if result.get("case_id") else ""
    return (
        "[CMTA survival prediction]\n"
        f"Cancer type: {result['cancer_type']}{patient_info}\n"
        f"Predicted survival time: {result['predicted_survival_months']:.1f} months\n"
        f"Risk level: {result['risk_level']}\n"
        f"Risk score: {float(result.get('risk_score', 0.0)):.4f}"
    )


if __name__ == "__main__":
    print("Testing CMTA survival prediction...")
    n_patches = 500
    pathomics = np.random.randn(n_patches, 1024).astype(np.float32)
    genomic_csv = os.getenv("CMTA_TEST_GENOMIC_CSV")
    if not genomic_csv:
        raise SystemExit("Set CMTA_TEST_GENOMIC_CSV=/path/to/single_patient.csv first.")

    for cancer in SUPPORTED_CANCER_TYPES:
        output = predict_survival(cancer, pathomics, genomic_features=genomic_csv)
        print(f"\n{cancer}:")
        print(format_survival_result(output))
