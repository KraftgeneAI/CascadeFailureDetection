"""
Tests for the scenario data contract.

Spec: specs/data-contract/spec.md. Every test names the acceptance criterion
(DC-xxx) it covers in its docstring.
"""

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from cascade_prediction import contract as C
from cascade_prediction.data.dataset import CascadeDataset
from cascade_prediction.data.generator.config import Settings
from contract_factory import make_scenario, make_timestep

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "specs" / "data-contract"


# ---------------------------------------------------------------------------
# Pre-contract reference implementations (copied verbatim in behaviour from
# dataset.py before the contract) — DC-021 / DC-022 require exact equality.
# ---------------------------------------------------------------------------
def _legacy_scada(raw, base_mva, isc):
    s = raw.astype(np.float32).copy()
    w = s.shape[1]
    s[:, 2] = s[:, 2] / base_mva
    s[:, 3] = s[:, 3] / base_mva
    s[:, 4] = s[:, 4] / base_mva
    if w >= 28:
        s[:, 23] = s[:, 23] / base_mva
        s[:, 24] = s[:, 24] / base_mva
        s[:, 25:28] = s[:, 25:28] / isc
    if s.shape[1] >= 22:
        s[:, 20] = s[:, 20] / base_mva
    elif s.shape[1] == 18:
        is_gen = (s[:, 11] == 1).astype(np.float32)
        pad = np.column_stack([np.zeros((s.shape[0], 3), np.float32), is_gen * 1.0]).astype(np.float32)
        s = np.concatenate([s, pad], axis=1)
    if s.shape[1] == 22:
        s = np.concatenate([s, np.zeros((s.shape[0], 6), np.float32)], axis=1)
    s[:, 12] = 0
    s[:, 13] = 0
    return s


def _legacy_edge(raw, base_mva):
    ea = raw.astype(np.float32).copy()
    w = ea.shape[1]
    if w >= 7:
        ea[:, 1] = ea[:, 1] / base_mva
        ea[:, 5] = ea[:, 5] / base_mva
        ea[:, 6] = ea[:, 6] / base_mva
    if w >= 10:
        ea[:, 8] = ea[:, 8] / base_mva
        ea[:, 9] = ea[:, 9] / base_mva
    if w >= 13:
        ea[:, 11] = ea[:, 11] / 16.0
        ea[:, 12] = ea[:, 12] / 30.0
    if w == 7:
        backfill = np.column_stack([np.ones(ea.shape[0], np.float32),
                                    ea[:, 1] * 1.22, ea[:, 1] * 1.47]).astype(np.float32)
        ea = np.concatenate([ea, backfill], axis=1)
    if ea.shape[1] == 10:
        ea = np.concatenate([ea, np.zeros((ea.shape[0], 3), np.float32)], axis=1)
    if ea.shape[1] == 13:
        ea = np.concatenate([ea, np.zeros((ea.shape[0], 1), np.float32)], axis=1)
    return ea


BASE_MVA = 1000.0
ISC = float(Settings.Interchange.INTERCHANGE_SCALE_MW)
CONSTS = {"INTERCHANGE_SCALE_MW": ISC}


def _contract_scada(raw):
    s = C.normalize_array(C.SCADA, C.migrate_array(C.SCADA, raw), base_mva=BASE_MVA,
                          constants=CONSTS)
    s[:, [C.SCADA.idx(n) for n in C.MODEL_MASKED_SCADA]] = 0
    return s


def _contract_edge(raw):
    return C.normalize_array(C.EDGE_ATTR, C.migrate_array(C.EDGE_ATTR, raw), base_mva=BASE_MVA,
                             constants=CONSTS)


def _write(path, scenario):
    with open(path, "wb") as f:
        pickle.dump([scenario], f)


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------
class TestDefinition:
    def test_arrays_are_fully_specified(self):
        """DC-001: every column has a name, unit, known norm policy and a range or category set."""
        for spec in C.ARRAYS:
            assert spec.width == len(spec.columns) > 0
            assert len(set(spec.names)) == spec.width
            for col in spec.columns:
                assert col.name and col.unit and col.description
                assert col.norm[0] in ("raw", "base_mva", "base_frequency", "divide")
                assert col.valid_range is not None or col.categories is not None, \
                    f"{spec.key}.{col.name} has no expected range"
                assert col.scope in ("local", "broadcast")

    def test_current_widths(self):
        """DC-001: the v1 layout widths."""
        assert (C.SCADA.width, C.PMU.width, C.EQUIPMENT.width,
                C.EDGE_ATTR.width, C.RAS_STATUS.width) == (28, 8, 10, 14, 3)

    def test_config_widths_derive_from_contract(self):
        """DC-002: config widths come from the contract."""
        assert Settings.Embedding.INFRA_SCADA_FEATURES == C.SCADA.width
        assert Settings.Embedding.INFRA_PMU_FEATURES == C.PMU.width
        assert Settings.Embedding.INFRA_EQUIPMENT_FEATURES == C.EQUIPMENT.width
        assert Settings.Embedding.NODE_FEATURE_BASE_DIM == C.NODE_FEATURE_BASE_DIM
        assert Settings.Embedding.NODE_FEATURE_DIM == C.NODE_FEATURE_DIM
        assert Settings.Model.EDGE_FEATURES == C.EDGE_ATTR.width
        assert Settings.RAS.RAS_FEATURES == C.RAS_STATUS.width

    def test_node_feature_layout(self):
        """DC-003: named offsets resolve to the documented layout."""
        assert C.NODE_FEATURE_BASE_DIM == 48 and C.NODE_FEATURE_DIM == 154
        assert C.base_index("power_injection") == 46
        assert C.base_index("reactive_injection") == 47
        assert C.base_index("scada", "loading_ratio") == 17
        assert C.base_index("pmu", "frequency") == 28 + 5
        assert C.node_feature_slice("t_pos") == slice(144, 145)
        with pytest.raises(C.ContractError):
            C.SCADA.idx("no_such_column")

    def test_layout_hash_ignores_ranges(self):
        """D6: LAYOUT_HASH covers what reaches the model, not validation ranges."""
        doc = C._layout_document()
        assert "valid_range" not in json.dumps(doc)
        assert len(C.LAYOUT_HASH) == 16


# ---------------------------------------------------------------------------
# Validation + producer
# ---------------------------------------------------------------------------
class TestValidation:
    def test_factory_scenario_is_clean(self):
        """DC-010/DC-050 baseline: an in-range scenario has no errors and no range violations."""
        rep = C.validate_scenario(make_scenario())
        assert rep.ok, rep.summary()
        assert rep.range_violation_count == 0, rep.summary()

    @pytest.mark.parametrize("mutate, needle", [
        (lambda s: s["sequence"][0].pop("edge_attr"), "edge_attr missing"),
        (lambda s: s["sequence"][1].__setitem__("scada_data", np.zeros((12, 22), np.float32)),
         "legacy layout"),
        (lambda s: s["sequence"][0]["pmu_sequence"].__setitem__((0, 0), np.nan), "NaN"),
        (lambda s: s["sequence"][0].__setitem__("node_labels", np.zeros(5, np.float32)), "length 5"),
        (lambda s: s["metadata"].pop("num_nodes"), "num_nodes"),
        (lambda s: s.__setitem__("edge_index", np.zeros((3, 20), np.int64)), "edge_index"),
    ])
    def test_structural_errors(self, mutate, needle):
        """DC-010: structural violations are errors."""
        sc = make_scenario()
        mutate(sc)
        rep = C.validate_scenario(sc)
        assert not rep.ok
        assert needle in rep.summary()
        with pytest.raises(C.ContractError):
            rep.raise_if_errors("test")

    def test_range_violations_are_counted_not_errors(self):
        """D4: out-of-range values are counted per column, not structural errors."""
        sc = make_scenario()
        sc["sequence"][0]["scada_data"][:, C.SCADA.idx("frequency")] = 63.0
        sc["sequence"][0]["edge_attr"][0, C.EDGE_ATTR.idx("branch_status")] = 0.5
        rep = C.validate_scenario(sc)
        assert rep.ok
        assert rep.range_violations["scada_data.frequency"] == 12
        assert rep.range_violations["edge_attr.branch_status"] == 1

    def test_generator_stamps_and_validates(self):
        """DC-010 / DC-011: generated scenarios are stamped and structurally valid."""
        pytest.importorskip("pypsa")
        from cascade_prediction.data.generator.simulator import PhysicsBasedGridSimulator
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        sc = sim.generate_scenario(stress_level=0.5, sequence_length=10)
        meta = sc["metadata"]
        assert meta["schema_version"] == C.SCHEMA_VERSION
        assert meta["layout_hash"] == C.LAYOUT_HASH
        assert isinstance(meta["contract_range_warnings"], int)
        assert C.validate_scenario(sc).ok

    @pytest.mark.xfail(strict=True, reason="ISSUE-FREQ-001: generator frequency pins at ~63 Hz "
                                           "on a 60 Hz system (spec §5)")
    def test_generated_data_has_no_range_violations(self):
        """DC-050: generated data is free of range violations."""
        pytest.importorskip("pypsa")
        from cascade_prediction.data.generator.simulator import PhysicsBasedGridSimulator
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        sc = sim.generate_scenario(stress_level=0.5, sequence_length=10)
        rep = C.validate_scenario(sc)
        assert rep.range_violation_count == 0, rep.summary()


# ---------------------------------------------------------------------------
# Migration + normalization equivalence
# ---------------------------------------------------------------------------
class TestMigrationAndNormalization:
    @pytest.mark.parametrize("width", [18, 22, 28])
    def test_scada_matches_pre_contract(self, width):
        """DC-021 / DC-022: SCADA migrate+normalize == old dataset padding+normalization."""
        rng = np.random.default_rng(width)
        raw = rng.uniform(-500, 500, (15, width)).astype(np.float32)
        raw[:, 11] = rng.choice([0.0, 1.0, 2.0], 15)
        np.testing.assert_array_equal(_contract_scada(raw), _legacy_scada(raw, BASE_MVA, ISC))

    @pytest.mark.parametrize("width", [7, 10, 13, 14])
    def test_edge_matches_pre_contract(self, width):
        """DC-021 / DC-022: edge_attr migrate+normalize == old dataset backfill+normalization."""
        rng = np.random.default_rng(width)
        raw = rng.uniform(-500, 500, (20, width)).astype(np.float32)
        np.testing.assert_allclose(_contract_edge(raw), _legacy_edge(raw, BASE_MVA), rtol=0, atol=1e-7)

    def test_unknown_width_is_rejected(self):
        """DC-021: only known legacy widths migrate."""
        with pytest.raises(C.ContractError):
            C.migrate_array(C.SCADA, np.zeros((3, 20), np.float32))

    def test_migration_is_logged(self, caplog):
        """DC-021: each migration is logged."""
        C._migration_logged.clear()
        with caplog.at_level(logging.WARNING, logger="cascade_prediction.contract"):
            C.migrate_array(C.EDGE_ATTR, np.zeros((4, 10), np.float32), context="t")
        assert "migrating legacy edge_attr from 10 to 14" in caplog.text


# ---------------------------------------------------------------------------
# Consumers: dataset strict/migrate + feature fixes
# ---------------------------------------------------------------------------
class TestDataset:
    def test_strict_accepts_stamped(self, tmp_path):
        """DC-020: stamped current-layout data loads in strict mode."""
        _write(tmp_path / "scenario_0.pkl", make_scenario(T=12))
        ds = CascadeDataset(str(tmp_path), mode="full_sequence")
        item = ds[0]
        assert item["scada_data"].shape[-1] == C.SCADA.width
        assert item["edge_attr"].shape[-1] == C.EDGE_ATTR.width
        assert item["node_features"].shape[-1] == C.NODE_FEATURE_DIM

    def test_strict_rejects_unstamped(self, tmp_path):
        """DC-020: unstamped, uncertified data fails at construction."""
        _write(tmp_path / "scenario_0.pkl", make_scenario(T=12, stamp=False))
        with pytest.raises(C.ContractError, match="not stamped"):
            CascadeDataset(str(tmp_path), mode="full_sequence")

    def test_strict_accepts_certified(self, tmp_path):
        """DC-042 / D5: certifying valid unstamped data makes strict mode accept it."""
        _write(tmp_path / "scenario_0.pkl", make_scenario(T=12, stamp=False))
        assert C.certify_directory(str(tmp_path)).ok
        manifest = json.loads((tmp_path / C.MANIFEST_NAME).read_text())
        assert manifest["layout_hash"] == C.LAYOUT_HASH
        ds = CascadeDataset(str(tmp_path), mode="full_sequence")
        assert ds[0]["scada_data"].shape[-1] == C.SCADA.width

    def test_certification_is_invalidated_by_edits(self, tmp_path):
        """DC-042: a file changed after certification is no longer certified."""
        f = tmp_path / "scenario_0.pkl"
        _write(f, make_scenario(T=12, stamp=False))
        C.certify_directory(str(tmp_path))
        assert C.is_certified(str(f))
        _write(f, make_scenario(T=13, stamp=False))
        os.utime(f, (1, 1))
        assert not C.is_certified(str(f))

    def test_certify_refuses_invalid_data(self, tmp_path):
        """DC-042: no manifest when any file is structurally invalid."""
        sc = make_scenario(stamp=False)
        sc["sequence"][0]["scada_data"] = np.zeros((12, 22), np.float32)
        _write(tmp_path / "scenario_0.pkl", sc)
        assert not C.certify_directory(str(tmp_path)).ok
        assert not (tmp_path / C.MANIFEST_NAME).exists()

    def test_migrate_mode_loads_legacy(self, tmp_path):
        """DC-021: migrate mode upgrades a legacy scenario; strict mode refuses it."""
        sc = make_scenario(T=12, stamp=False)
        for ts in sc["sequence"]:
            ts["scada_data"] = ts["scada_data"][:, :18].copy()
            ts["edge_attr"] = ts["edge_attr"][:, :7].copy()
            ts.pop("ras_status")
        _write(tmp_path / "scenario_0.pkl", sc)
        item = CascadeDataset(str(tmp_path), mode="full_sequence", contract_mode="migrate")[0]
        assert item["scada_data"].shape[-1] == C.SCADA.width
        assert item["edge_attr"].shape[-1] == C.EDGE_ATTR.width
        assert item["ras_status"].shape[-1] == C.RAS_STATUS.width
        with pytest.raises(C.ContractError):
            CascadeDataset(str(tmp_path), mode="full_sequence")

    def test_t_pos_is_masked_by_name(self, tmp_path):
        """D7 / DC-003: t_pos is zeroed; the cap-bank delta2 column is no longer clobbered."""
        sc = make_scenario(T=12)
        cap = C.SCADA.idx("cap_bank_status")
        for t, ts in enumerate(sc["sequence"]):
            ts["scada_data"][:, cap] = 1.0 if t in (4, 5) else 0.0   # delta2 != 0 at t=4..7
        _write(tmp_path / "scenario_0.pkl", sc)
        ds = CascadeDataset(str(tmp_path), mode="full_sequence", fixed_window=(0, 12))
        nf = ds[0]["node_features"]
        assert torch.all(nf[:, :, C.node_feature_slice("t_pos")] == 0)
        d2_cap = C.node_feature_slice("delta2").start + C.base_index("scada", "cap_bank_status")
        assert d2_cap == 114                      # the index the old hard-coded mask hit
        assert torch.any(nf[:, :, d2_cap] != 0)   # ...and it now carries signal

    def test_susceptibility_uses_power_injection(self):
        """DC-003: cascade_initiation_risk is |P_inj| / sum(adjacent limits), not equipment age."""
        N, E, T = 3, 2, 2
        edge_index = np.array([[0, 1], [1, 2]])
        def ts():
            t = make_timestep(N, E)
            ea = t["edge_attr"]
            ea[:, C.EDGE_ATTR.idx("thermal_limit")] = 100.0
            ea[:, C.EDGE_ATTR.idx("p_flow")] = 10.0
            t["power_injection"] = np.array([50.0, -20.0, 5.0], np.float32)
            t["equipment_status"][:, C.EQUIPMENT.idx("equipment_age")] = 30.0
            return t
        seq = [ts() for _ in range(T)]
        to_t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        nf = CascadeDataset._build_node_features(
            scada_list=[to_t(s["scada_data"]) for s in seq],
            pmu_list=[to_t(s["pmu_sequence"]) for s in seq],
            equip_list=[to_t(s["equipment_status"]) for s in seq],
            sequence=seq, num_nodes=N, edge_index=edge_index,
            edge_attr_list=[s["edge_attr"] for s in seq])
        f2 = nf[-1, :, C.node_feature_slice("susceptibility").start + 1]
        # Edges are attributed to their destination node: node 0 has none,
        # node 1 has edge 0->1, node 2 has edge 1->2 (100 MVA each).
        # With the pre-contract bug node 1 would read equipment_age: 30/100 = 0.3.
        assert f2[0].item() == pytest.approx(0.0, abs=1e-6)
        assert f2[1].item() == pytest.approx(20 / 100, abs=1e-4)
        assert f2[2].item() == pytest.approx(5 / 100, abs=1e-4)


# ---------------------------------------------------------------------------
# Trainer targets + checkpoints
# ---------------------------------------------------------------------------
class TestTrainerAndCheckpoints:
    def test_physics_targets_by_name(self):
        """DC-023: voltage / temperature / flow targets come from the named columns."""
        from cascade_prediction.training.trainer import Trainer
        B, T, N, E = 2, 3, 4, 5
        scada = torch.randn(B, T, N, C.SCADA.width)
        edge = torch.randn(B, T, E, C.EDGE_ATTR.width)
        batch = {"node_failure_labels": torch.zeros(B, N), "scada_data": scada, "edge_attr": edge}
        tg = Trainer._prepare_targets(None, batch)
        assert torch.equal(tg["physics_voltage_target"], scada[:, -1, :, C.SCADA.idx("voltage")])
        assert torch.equal(tg["physics_temp_target"], scada[:, -1, :, C.SCADA.idx("equipment_temp")])
        assert torch.equal(tg["physics_flow_target"], edge[:, -1, :, C.EDGE_ATTR.idx("p_flow")])

    def test_saved_checkpoint_is_stamped(self, tmp_path):
        """DC-030: trainer checkpoints carry schema_version + layout_hash."""
        from cascade_prediction.training.trainer import Trainer
        model = torch.nn.Linear(2, 2)
        tr = Trainer(model, [], [], criterion=None, device=torch.device("cpu"),
                     output_dir=str(tmp_path))
        tr.save_checkpoint("ck.pth", epoch=0)
        ck = torch.load(tmp_path / "ck.pth", weights_only=False)
        assert ck["layout_hash"] == C.LAYOUT_HASH
        assert ck["schema_version"] == C.SCHEMA_VERSION

    def test_checkpoint_mismatch_is_refused(self, caplog):
        """DC-031: a different layout hash raises; an unstamped checkpoint warns."""
        with pytest.raises(C.ContractError, match="Retrain"):
            C.check_checkpoint({"layout_hash": "0000000000000000", "schema_version": "0.9.0"})
        C.check_checkpoint(C.contract_stamp())  # current: fine
        with caplog.at_level(logging.WARNING, logger="cascade_prediction.contract"):
            C.check_checkpoint({"model_state_dict": {}})
        assert "no data-contract stamp" in caplog.text


# ---------------------------------------------------------------------------
# Tooling
# ---------------------------------------------------------------------------
class TestTooling:
    def test_committed_reference_is_current(self):
        """DC-040: the committed generated reference matches the contract."""
        committed = json.loads((SPEC_DIR / "data_contract.json").read_text(encoding="utf-8"))
        assert committed == json.loads(json.dumps(C.to_document())), \
            "Run: python -m cascade_prediction.contract export specs/data-contract"
        assert (SPEC_DIR / "data_contract.md").read_text(encoding="utf-8") == C.to_markdown()

    def test_validate_cli(self, tmp_path, capsys):
        """DC-041: the validate command reports structural errors with a non-zero exit."""
        good, bad = make_scenario(stamp=False), make_scenario(stamp=False)
        bad["sequence"][0].pop("pmu_sequence")
        _write(tmp_path / "scenario_0.pkl", good)
        _write(tmp_path / "scenario_1.pkl", bad)
        assert C.main(["validate", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "scenario_1.pkl" in out and "pmu_sequence missing" in out

    def test_validate_cli_fail_on_ranges(self, tmp_path):
        """DC-041 / D4: --fail-on-ranges turns range warnings into a failing exit code."""
        sc = make_scenario(stamp=False)
        sc["sequence"][0]["scada_data"][:, C.SCADA.idx("frequency")] = 63.0
        _write(tmp_path / "scenario_0.pkl", sc)
        assert C.main(["validate", str(tmp_path)]) == 0
        assert C.main(["validate", str(tmp_path), "--fail-on-ranges"]) == 1
