# Spec: Scenario Data Contract

- **Status:** Accepted, v1.0.0
- **Owner:** CascadeFailureDetection
- **Code (source of truth):** `cascade_prediction/contract.py`
- **Generated reference:** `specs/data-contract/data_contract.json` and
  `data_contract.md` (both produced by
  `python -m cascade_prediction.contract export specs/data-contract`; never
  edit them by hand)
- **Constitution rules:** 4, 5, 6, 10

## 1. Problem

The generator, dataset, trainer, loss, inference and backend each hard-code
column positions and widths for the same arrays (SCADA 28 columns, edge_attr
14, and so on). Every time an array was widened, some of those hard-coded
numbers were missed, and nothing failed. Known casualties, all fixed by this
spec:

- **Susceptibility feature F2 used the wrong input.** `dataset.py` read "power
  injection" from base feature 36. After SCADA grew to 28 columns, 36 is
  `equipment_age`; power injection is at 46.
- **The time-position mask moved.** `node_features[:,:,114]=0` was written to
  hide `t_pos` when the base layout had 38 features. After the widening, 114
  pointed at the 2-step delta of the capacitor-bank status, so `t_pos` reached
  the model.
- **A fallback read the wrong column.** The reactive-injection fallback used
  SCADA col 5 (equipment temperature) as reactive load.
- **Old checkpoints loaded without complaint.** A checkpoint trained on an
  older layout gave no error, because nothing recorded which layout it was
  trained on.

## 2. Scope

**In scope:** every per-timestep array a scenario carries (`scada_data`,
`pmu_sequence`, `equipment_status`, `edge_attr`, `ras_status`, and the 1-D
label and injection vectors), the scenario envelope (`sequence`, `edge_index`,
`metadata`), the model's node-feature layout, the normalization policy for
each column, legacy-layout migration, and version stamps on scenarios and
checkpoints.

**Out of scope for v1:** the topology block schema (`topology`), the
`cascade_events` payload, the API contract, and the agent contract.

## 3. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Contract lives in `cascade_prediction/contract.py` with no imports from the package. `config.py` derives its widths from the contract, not the other way round. | One source of truth; no import cycles. |
| D2 | Columns are addressed by **name**; indices are derived. | Appending or reordering columns can't silently shift consumers. |
| D3 | **Training rejects** any data that isn't stamped with the current layout (or certified, see D5). **Non-training consumers** (backend, inference) migrate legacy layouts through one explicit, logged function. | Constitution rule 10; keeps the demo working with old scenarios. |
| D4 | Structural checks (keys, shapes, widths, dtype, finiteness) **fail hard everywhere**. Value-range checks **warn at generation** and are counted in scenario metadata; **tests and CI fail** on any range violation. | Generation shouldn't abort on a physics edge case, but no violation can reach a merged branch. |
| D5 | Existing unstamped data that passes every structural check can be **certified** with `python -m cascade_prediction.contract certify <dir>`. This writes `contract_manifest.json` next to the files; strict mode accepts certified files. | Avoids regenerating valid data just to add a stamp. |
| D6 | Versioning: `SCHEMA_VERSION` is semver. MAJOR = any change to array layout, column meaning or normalization. MINOR = new validation rules or ranges. PATCH = docs only. `LAYOUT_HASH` hashes only what reaches the model (names, order, units, normalization), so changing a range doesn't invalidate checkpoints. | Checkpoints should be invalidated by exactly the changes that break them. |
| D7 | `t_pos` is masked (zeroed) in the model input **by name**, along with SCADA `time_ratio` and `stress_level`. | Restores the mask's original intent (decided 2026-09-23). |

## 4. Acceptance criteria

**Definition**

- **DC-001:** `contract.py` defines every in-scope array with ordered, named
  columns, each with a unit, a normalization policy and an expected value
  range. Array widths are derived from the column lists.
- **DC-002:** `Settings.Embedding.INFRA_SCADA_FEATURES`,
  `NODE_FEATURE_BASE_DIM`, `NODE_FEATURE_DIM` and `Settings.Model.EDGE_FEATURES`
  are derived from the contract.
- **DC-003:** The node-feature layout (base, deltas, `t_pos`, TTF,
  susceptibility) is defined in the contract; the dataset looks up every
  offset by name.

**Producer (generator)**

- **DC-010:** Every generated timestep passes structural validation, or
  generation raises `ContractError`.
- **DC-011:** Every generated scenario's metadata carries `schema_version`,
  `layout_hash` and `contract_range_warnings` (a count).
- **DC-012:** `utils.validate_scenario` delegates to the contract.

**Consumers**

- **DC-020:** In `strict` mode (the training default), `CascadeDataset`
  raises `ContractError` on a scenario that is neither stamped with the
  current `layout_hash` nor certified.
- **DC-021:** In `migrate` mode, legacy layouts (SCADA 18/22 columns;
  edge_attr 7/10/13 columns) are brought up to the current layout by
  `contract.migrate_array`, and each migration is logged. Migrated values are
  identical to the pre-contract padding behavior.
- **DC-022:** Normalization is driven by the per-column policy in the
  contract, and its output is numerically identical to the pre-contract
  implementation.
- **DC-023:** Trainer physics targets (voltage, temperature, line flow) are
  looked up by column name.
- **DC-024:** The backend `ScenarioService` reads SCADA and edge fields by
  name and uses migrate mode.

**Artifacts**

- **DC-030:** Every checkpoint the trainer saves includes `schema_version`
  and `layout_hash`.
- **DC-031:** Loading a checkpoint whose `layout_hash` doesn't match the
  current contract raises `ContractError`. Loading an unstamped checkpoint
  logs a warning.

**Tooling**

- **DC-040:** `python -m cascade_prediction.contract export <dir>` writes the
  JSON schema and the Markdown reference. CI fails if the committed copies
  differ from the output.
- **DC-041:** `python -m cascade_prediction.contract validate <dir>` reports
  structural errors and range warnings for every scenario file in `<dir>`.
- **DC-042:** `certify <dir>` writes `contract_manifest.json` only when every
  file passes structural validation.

**Data quality (ranges)**

- **DC-050:** Generated data has zero range violations. *(Currently fails;
  see §5.)*

## 5. Known issues surfaced by the contract

- **ISSUE-FREQ-001:** System frequency in generated data climbs from about
  62.6 Hz to a flat 63.0 Hz on a 60 Hz system, in every sampled scenario, with
  generation about 200 MW above load. SCADA column `frequency` has an expected
  range of 57.0–61.8 Hz, so every current scenario violates DC-050. The
  generator physics fix is tracked separately. DC-050's test is marked
  `xfail(strict=True)` against this issue, so it starts failing (and must be
  un-xfailed) as soon as the fix lands.
- **Observed, not a violation:** storage MW (SCADA 23) is in MW but only
  reaches a few tenths of a MW because of the 0.2 MW/Hz droop, so it carries
  almost no signal after base-MVA normalization. Edge susceptance and
  conductance (edge cols 3–4) are all zero in current data.

## 6. Adopting on existing data

Data generated before the contract has the current layout but no stamp, so
strict (training) mode rejects it. Certify each split once:

```bash
python -m cascade_prediction.contract certify data/train
python -m cascade_prediction.contract certify data/val
python -m cascade_prediction.contract certify data/test
```

Certification covers exactly the files present at that moment: a file added
or modified afterwards is rejected until you certify again. Newly generated
data is stamped automatically.

Verified against three real validation scenarios: after certification the
dataset output is bit-identical to the pre-contract pipeline except for the
three intentional fixes in §1 (node features 114, 144 and 150). Any
checkpoint trained before this change is therefore stale; retrain.

## 7. Change procedure

1. Edit the column lists in `contract.py`, and bump `SCHEMA_VERSION`
   following D6.
2. Run `python -m cascade_prediction.contract export specs/data-contract`
   and commit the regenerated reference.
3. Update this spec if a decision or criterion changes.
4. A MAJOR bump means: regenerate (or migrate and re-certify) the data and
   retrain. Old checkpoints will refuse to load.
