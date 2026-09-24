# Data contract reference

Generated from `cascade_prediction/contract.py`. Do not edit by hand. Schema **1.0.0**, layout hash `62c1a44b78b43ec1`.

## `scada_data` (N x 28)

Per-node SCADA measurements. Broadcast columns hold one system-wide value repeated on every node.

Legacy widths migratable outside training: 18, 22.

| # | name | unit | norm | expected | scope | since | description |
|---|------|------|------|----------|-------|-------|-------------|
| 0 | `voltage` | p.u. | raw | [0, 1.5] | local | base | Bus voltage magnitude (0 = de-energized) |
| 1 | `angle` | rad | raw | [-3.14159, 3.14159] | local | base | Bus voltage angle |
| 2 | `generation` | MW | base_mva | [-1000, 20000] | local | base | Active generation incl. storage injection |
| 3 | `node_reactive` | MVAr | base_mva | [-10000, 10000] | local | base | Net reactive power at the bus |
| 4 | `load` | MW | base_mva | [0, 10000] | local | base | Active load |
| 5 | `equipment_temp` | degC | raw | [-40, 250] | local | base | Equipment temperature (physics temperature target) |
| 6 | `frequency` | Hz | raw | [57, 61.8] | broadcast | base | System frequency (raw Hz; not normalized in SCADA) |
| 7 | `equipment_age` | years | raw | [0, 100] | local | base | Equipment age |
| 8 | `equipment_condition` | fraction | raw | [0, 1] | local | base | Condition index, 1 = new |
| 9 | `gen_capacity` | MW | raw | [0, 50000] | local | base | Installed generation capacity |
| 10 | `base_load` | MW | raw | [0, 10000] | local | base | Nominal (base) load |
| 11 | `node_type` | category | raw | {0, 1, 2} | local | base | 0 = load, 1 = generator, 2 = substation |
| 12 | `time_ratio` | fraction | raw | [0, 1] | broadcast | base | t / sequence_length (masked in model input) |
| 13 | `stress_level` | fraction | raw | [0, 1] | broadcast | base | Scenario stress level (masked in model input) |
| 14 | `voltage_ratio` | ratio | raw | [0, 2] | local | base | voltage / voltage_failure_threshold (<1 = danger) |
| 15 | `temp_ratio` | ratio | raw | [0, 3] | local | base | temp / temperature_failure_threshold (>1 = danger) |
| 16 | `freq_ratio` | ratio | raw | [0.9, 1.1] | local | base | frequency / frequency_failure_threshold (<1 = danger) |
| 17 | `loading_ratio` | ratio | raw | [0, 5] | local | base | load / base_load / loading_failure_threshold (>1 = danger) |
| 18 | `cap_bank_status` | binary | raw | {0, 1} | local | phase3 | Capacitor bank in service |
| 19 | `reactor_status` | binary | raw | {0, 1} | local | phase3 | Shunt reactor in service |
| 20 | `svc_q` | MVAr | base_mva | [-1000, 1000] | local | phase3 | SVC reactive output |
| 21 | `avr_setpoint` | p.u. | raw | [0, 1.2] | local | phase3 | AVR voltage setpoint (0 on non-generator buses) |
| 22 | `storage_soc` | fraction | raw | [0, 1] | local | phase5 | Storage state of charge (0 on non-storage buses) |
| 23 | `storage_mw` | MW | base_mva | [-500, 500] | local | phase5 | Storage active power, + = discharging |
| 24 | `storage_mvar` | MVAr | base_mva | [-500, 500] | local | phase5 | Storage reactive power |
| 25 | `actual_interchange` | MW | divide:INTERCHANGE_SCALE_MW | [-20000, 20000] | broadcast | phase5 | Net actual interchange (sum of signed tie flows) |
| 26 | `scheduled_interchange` | MW | divide:INTERCHANGE_SCALE_MW | [-20000, 20000] | broadcast | phase5 | Scheduled interchange |
| 27 | `ace` | MW | divide:INTERCHANGE_SCALE_MW | [-20000, 20000] | broadcast | phase5 | Area control error = schedule - actual |

## `pmu_sequence` (N x 8)

Per-node PMU measurements.

| # | name | unit | norm | expected | scope | since | description |
|---|------|------|------|----------|-------|-------|-------------|
| 0 | `voltage` | p.u. | raw | [0, 1.5] | local | base | Bus voltage magnitude |
| 1 | `angle` | rad | raw | [-3.14159, 3.14159] | local | base | Bus voltage angle |
| 2 | `generation` | MW | raw | [-1000, 20000] | local | base | Active generation |
| 3 | `load` | MW | raw | [0, 10000] | local | base | Active load |
| 4 | `equipment_temp` | degC | raw | [-40, 250] | local | base | Equipment temperature |
| 5 | `frequency` | Hz | base_frequency | [57, 61.8] | broadcast | base | System frequency |
| 6 | `mean_line_loading` | ratio | raw | [0, 5] | broadcast | base | Mean line loading ratio across the grid |
| 7 | `node_reactive` | MVAr | raw | [-10000, 10000] | local | base | Net reactive power at the bus |

## `equipment_status` (N x 10)

Per-node equipment health and thermal state.

| # | name | unit | norm | expected | scope | since | description |
|---|------|------|------|----------|-------|-------|-------------|
| 0 | `equipment_age` | years | raw | [0, 100] | local | base | Equipment age |
| 1 | `equipment_condition` | fraction | raw | [0, 1] | local | base | Condition index |
| 2 | `equipment_temp` | degC | raw | [-40, 250] | local | base | Equipment temperature |
| 3 | `thermal_capacity` | ratio | raw | [0, 5] | local | base | Thermal capacity factor |
| 4 | `cooling_effectiveness` | fraction | raw | [0, 1] | local | base | Cooling effectiveness |
| 5 | `thermal_time_constant` | x30 min | raw | [0, 10] | local | base | Thermal time constant / 30 |
| 6 | `temp_ratio` | ratio | raw | [0, 3] | local | base | temp / temperature_failure_threshold |
| 7 | `node_type` | category | raw | {0, 1, 2} | local | base | 0 = load, 1 = generator, 2 = substation |
| 8 | `gen_capacity_fraction` | fraction | raw | [0, 1] | local | base | gen_capacity / max gen_capacity |
| 9 | `load_ratio` | ratio | raw | [0, 5] | local | base | load / base_load |

## `edge_attr` (E x 14)

Per-branch attributes. Mirrored rows of one undirected branch share values.

Legacy widths migratable outside training: 7, 10, 13.

| # | name | unit | norm | expected | scope | since | description |
|---|------|------|------|----------|-------|-------|-------------|
| 0 | `reactance` | ohm | raw | [0, 1000] | local | base | Series reactance |
| 1 | `thermal_limit` | MVA | base_mva | [0, 100000] | local | base | Normal rating, derated for the step |
| 2 | `resistance` | ohm | raw | [0, 1000] | local | base | Series resistance |
| 3 | `susceptance` | S | raw | [-1000, 1000] | local | base | Shunt susceptance |
| 4 | `conductance` | S | raw | [-1000, 1000] | local | base | Shunt conductance |
| 5 | `p_flow` | MW | base_mva | [-100000, 100000] | local | base | Active power flow (physics flow target) |
| 6 | `q_flow` | MVAr | base_mva | [-100000, 100000] | local | base | Reactive power flow |
| 7 | `branch_status` | category | raw | {-1, 0, 1} | local | phase1 | 1 = closed, 0 = forced open, -1 = planned open |
| 8 | `emergency_limit` | MVA | base_mva | [0, 100000] | local | phase1 | Emergency (LEP) rating, derated |
| 9 | `load_dump_limit` | MVA | base_mva | [0, 100000] | local | phase1 | Load-dump rating, derated |
| 10 | `branch_type` | category | raw | {0, 1, 2} | local | phase2 | 0 = line, 1 = LTC transformer, 2 = phase shifter |
| 11 | `tap_position` | steps | divide:16.0 | [-16, 16] | local | phase2 | LTC tap position |
| 12 | `tap_angle` | deg | divide:30.0 | [-30, 30] | local | phase2 | Phase-shifter angle |
| 13 | `is_tie_line` | binary | raw | {0, 1} | local | phase5 | Branch crosses the boundary to the neighbouring RC |

## `ras_status` (S x 3)

Per-scheme Remedial Action Scheme telemetry.

| # | name | unit | norm | expected | scope | since | description |
|---|------|------|------|----------|-------|-------|-------------|
| 0 | `arming_status` | category | raw | {0, 1, 2, 3} | local | phase4 | 0 = available, 1 = armed, 2 = triggered, 3 = operated |
| 1 | `monitored_loading` | ratio | raw | [0, 5] | local | phase4 | Loading of the scheme's monitored line |
| 2 | `action_type` | category | raw | {0, 1, 2} | local | phase4 | 0 = trip generator, 1 = shed load, 2 = open line |

## Per-timestep vectors

| key | rows | unit | description |
|---|---|---|---|
| `node_labels` | N | binary | 1 if the node has failed by this step |
| `cascade_timing` | N | steps | Failure time relative to cascade start; -1 = does not fail |
| `power_injection` | N | MW | generation - load |
| `reactive_injection` | N | MVAr | Net reactive injection |
| `conductance` | E | S | Branch conductance (mirror of edge_attr) |
| `susceptance` | E | S | Branch susceptance (mirror of edge_attr) |
| `thermal_limits` | E | MVA | Static normal rating (not derated) |

## Model node-feature layout

Base dim 48, total dim 154.

| block | offset | width |
|---|---|---|
| base | 0 | 48 |
| delta1 | 48 | 48 |
| delta2 | 96 | 48 |
| t_pos | 144 | 1 |
| ttf | 145 | 4 |
| susceptibility | 149 | 5 |

Base parts: scada@0, pmu@28, equipment@36, power_injection@46, reactive_injection@47.

Masked in model input: SCADA `time_ratio`, `stress_level`; node features `t_pos`.
