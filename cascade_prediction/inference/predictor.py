"""
CascadePredictor
================
Inference service for cascade failure prediction.
Loads a trained UnifiedCascadePredictionModel and runs prediction
on a single scenario using the same CascadeDataset pipeline as training.
"""

import pickle
import torch
import numpy as np
from typing import Dict, List

from ..models.unified_model import UnifiedCascadePredictionModel
from ..data.dataset import CascadeDataset
from ..data.collation import collate_cascade_batch
from ..data.generator.config import Settings


class CascadePredictor:
    def __init__(
        self,
        model_path: str,
        topology_path: str = None,
        device: torch.device = None,
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_freq: float = Settings.Dataset.BASE_FREQUENCY,
    ):
        self.device = device or torch.device("cpu")
        self.base_mva = base_mva
        self.base_freq = base_freq

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = UnifiedCascadePredictionModel(
            embedding_dim=Settings.Model.EMBEDDING_DIM,
            hidden_dim=Settings.Model.HIDDEN_DIM,
            num_gnn_layers=Settings.Model.NUM_GNN_LAYERS,
            heads=Settings.Model.HEADS,
            dropout=Settings.Model.GAT_DROPOUT,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.cascade_threshold = checkpoint.get("cascade_threshold", Settings.Training.CASCADE_THRESHOLD)
        self.node_threshold = checkpoint.get("node_threshold", Settings.Training.NODE_THRESHOLD)
        print(f"✓ Model loaded. Thresholds: Cascade={self.cascade_threshold:.2f}, Node={self.node_threshold:.2f}")

    def predict_scenario(
        self,
        data_path: str,
        scenario_idx: int,
        window_size: int = Settings.Simulation.DEFAULT_SEQUENCE_LENGTH,
        batch_size: int = Settings.Training.BATCH_SIZE,
    ) -> Dict:
        dataset = CascadeDataset(
            data_path,
            mode="full_sequence",
            base_mva=self.base_mva,
            base_frequency=self.base_freq,
        )

        print(f"Running inference on scenario {scenario_idx} / {len(dataset) - 1}...")
        item = dataset[scenario_idx]
        if not item:
            raise ValueError(f"Scenario {scenario_idx} could not be loaded from {data_path}")

        original_seq_len = int(item.get("original_sequence_length", Settings.Simulation.DEFAULT_SEQUENCE_LENGTH))

        batch = collate_cascade_batch([item])
        batch_dev = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = self.model(batch_dev)

        probs = outputs["failure_probability"].squeeze(-1).squeeze(0).cpu().numpy()  # [N]

        DT_MINUTES = Settings.Thermal.DT_MINUTES
        pred_timing_normed = outputs["cascade_timing"].squeeze(-1).squeeze(0).cpu().numpy()  # [N]
        pred_timing_minutes = pred_timing_normed * original_seq_len * DT_MINUTES

        final_risk_scores = outputs["risk_scores"][0].mean(dim=0).cpu().numpy().tolist()
        final_sys_state = {
            "frequency": float(outputs["frequency"].mean().item()),
            "voltages": outputs["voltages"][0].reshape(-1).cpu().numpy().tolist(),
        }

        max_probs = {n: float(probs[n]) for n in range(len(probs))}
        risky_nodes = [n for n, p in max_probs.items() if p > self.node_threshold]

        ranked_nodes = sorted(
            [
                {
                    "node_id": n,
                    "score": max_probs[n],
                    "pred_time_minutes": float(pred_timing_minutes[n]),
                }
                for n in risky_nodes
            ],
            key=lambda x: x["pred_time_minutes"],
        )

        cascade_path = self._build_cascade_path(ranked_nodes)
        ground_truth = self._load_ground_truth(dataset, scenario_idx)

        return {
            "inference_time": 0.0,
            "cascade_detected": bool(risky_nodes),
            "cascade_probability": ranked_nodes[0]["score"] if ranked_nodes else 0.0,
            "ground_truth": ground_truth,
            "high_risk_nodes": risky_nodes,
            "risk_assessment": final_risk_scores,
            "top_nodes": ranked_nodes,
            "cascade_path": cascade_path,
            "system_state": final_sys_state,
        }

    def _build_cascade_path(self, ranked_nodes: List[Dict]) -> List[Dict]:
        cascade_path = []
        if not ranked_nodes:
            return cascade_path
        current_rank = 1
        last_time = ranked_nodes[0]["pred_time_minutes"]
        for node in ranked_nodes:
            if node["pred_time_minutes"] > last_time + 0.5:
                current_rank += 1
                last_time = node["pred_time_minutes"]
            cascade_path.append({
                "order": current_rank,
                "node_id": node["node_id"],
                "ranking_score": node["score"],
                "pred_time_minutes": node["pred_time_minutes"],
            })
        return cascade_path

    def _load_ground_truth(self, dataset: CascadeDataset, scenario_idx: int) -> Dict:
        with open(dataset.scenario_files[scenario_idx], "rb") as f:
            raw = pickle.load(f)
        meta = (raw[0] if isinstance(raw, list) else raw).get("metadata", {})

        GT_DT = Settings.Thermal.DT_MINUTES
        gt_path = []
        if "failed_nodes" in meta and "failure_times" in meta:
            gt_path = sorted(
                [
                    {"node_id": int(n), "time_minutes": float(t) * GT_DT}
                    for n, t in zip(meta["failed_nodes"], meta["failure_times"])
                ],
                key=lambda x: x["time_minutes"],
            )

        return {
            "is_cascade": meta.get("is_cascade"),
            "failed_nodes": meta.get("failed_nodes", []),
            "cascade_path": gt_path,
            "ground_truth_risk": meta.get("ground_truth_risk", []),
        }
