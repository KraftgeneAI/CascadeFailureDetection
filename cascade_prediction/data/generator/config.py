"""
config.py — Central configuration for the cascade failure detection simulator.

All hardcoded numeric constants live here. Import with:
    from cascade_prediction.data.generator.config import (
        PowerSystemConfig, FrequencyConfig, ThermalConfig, ...
    )
"""

# ---------------------------------------------------------------------------
# Power System Base
# ---------------------------------------------------------------------------
class PowerSystemConfig:
    SN_MVA        = 1000.0   # System base MVA. Load buses are 30–200 MW so 100 MVA
                             # gives 80+ pu total load — NR diverges from flat start.
                             # 1000 MVA keeps per-bus loads at 0.03–0.20 pu.
    V_NOM_KV      = 138.0    # Nominal bus voltage (kV)
    BASE_FREQUENCY = 60.0    # Nominal system frequency (Hz)

    # Per-unit base impedance: Z_base = V_nom² / S_base  (190.44 Ω)
    Z_BASE = V_NOM_KV ** 2 / SN_MVA


# ---------------------------------------------------------------------------
# AC Power Flow (PyPSA)
# ---------------------------------------------------------------------------
class PowerFlowConfig:
    SLACK_V_SET        = 1.0            # Slack bus voltage setpoint (p.u.)
    PV_V_SET_MIN       = 0.98           # PV generator voltage schedule min (p.u.)
    PV_V_SET_MAX       = 1.02           # PV generator voltage schedule max (p.u.)
    Q_LOAD_FACTOR      = 0.30           # Reactive load factor (power-factor correction, 0–1)
    VOLTAGE_COLLAPSE_PROXY = 0.85       # Multiplier applied to last stable voltages on divergence


# ---------------------------------------------------------------------------
# Line Impedance
# ---------------------------------------------------------------------------
class LineImpedanceConfig:
    # Reactance (p.u.) — base + distance-proportional term, then clipped
    X_PU_BASE_MIN   = 0.03
    X_PU_BASE_MAX   = 0.08
    X_PU_DIST_MIN   = 0.05
    X_PU_DIST_MAX   = 0.10
    X_PU_CLIP_MIN   = 0.02
    X_PU_CLIP_MAX   = 0.20

    # Resistance as fraction of reactance (R/X ratio)
    R_X_RATIO_MIN   = 0.05
    R_X_RATIO_MAX   = 0.15

    # Shunt susceptance (p.u.) — before dividing by Z_base
    B_PU_MIN        = 0.5e-4
    B_PU_MAX        = 2.0e-4

    # Thermal limit multipliers by line distance (fraction of max system load)
    THERMAL_SHORT_DIST_KM   = 30        # km — short/medium boundary
    THERMAL_MEDIUM_DIST_KM  = 60        # km — medium/long boundary
    THERMAL_SHORT_MIN       = 1.5
    THERMAL_SHORT_MAX       = 2.5
    THERMAL_MEDIUM_MIN      = 1.0
    THERMAL_MEDIUM_MAX      = 1.8
    THERMAL_LONG_MIN        = 0.8
    THERMAL_LONG_MAX        = 1.5
    THERMAL_MARGIN_MIN      = 1.5       # Convergence margin multiplier
    THERMAL_MARGIN_MAX      = 2.0
    THERMAL_MIN_FRACTION    = 0.05      # Floor: 5% of total load per line

    # Flow-proportional rating calibration (fixes the uniform-566MW limits:
    # the distance heuristic above was flattened by the floor and its
    # avg-flow base double-counted directed edges, so backbone corridors
    # carrying 1400+ MW got the same rating as 20 MW spurs → load-dump
    # trips at t=0 even in NORMAL scenarios).
    # Each line's NORMAL rating = base-case |S| × U(HEADROOM_MIN, MAX),
    # floored at THERMAL_FLOOR_MVA for lightly-loaded lines. Nominal loading
    # then sits at 1/1.8–1/1.3 ≈ 0.55–0.77 — overloads must EMERGE from the
    # stress ramp and failures, not exist at birth.
    THERMAL_HEADROOM_MIN    = 1.30
    THERMAL_HEADROOM_MAX    = 1.80
    THERMAL_FLOOR_MVA       = 60.0


# ---------------------------------------------------------------------------
# Under-Frequency Load Shedding (UFLS)
# ---------------------------------------------------------------------------
class UFLSConfig:
    STAGES = [
        {"frequency": 59.3, "load_shed": 0.10},   # Stage 1 — shed 10 %
        {"frequency": 59.0, "load_shed": 0.15},   # Stage 2 — shed 15 %
        {"frequency": 58.7, "load_shed": 0.20},   # Stage 3 — shed 20 %
    ]


# ---------------------------------------------------------------------------
# Frequency Dynamics
# ---------------------------------------------------------------------------
class FrequencyConfig:
    BASE_FREQUENCY  = PowerSystemConfig.BASE_FREQUENCY

    # Generator inertia H constants (seconds), by capacity class
    INERTIA_LARGE_MW_THRESHOLD  = 400   # MW — large generator threshold
    INERTIA_MEDIUM_MW_THRESHOLD = 150   # MW — medium generator threshold
    INERTIA_LARGE_MIN   = 4.0
    INERTIA_LARGE_MAX   = 6.0
    INERTIA_MEDIUM_MIN  = 2.5
    INERTIA_MEDIUM_MAX  = 4.0
    INERTIA_SMALL_MIN   = 1.5
    INERTIA_SMALL_MAX   = 2.5

    # Damping
    DAMPING_MIN     = 1.0
    DAMPING_MAX     = 2.0
    D_TOTAL         = 0.04              # Combined droop + load damping (p.u./p.u.)

    # Governor response
    TAU_GOV_MIN     = 0.5               # Governor time constant (minutes)

    # Frequency operating envelope
    FREQ_MIN_HZ     = 55.0
    FREQ_MAX_HZ     = 65.0
    DELTA_F_MAX_HZ  = 3.0              # Max steady-state deviation before collapse


# ---------------------------------------------------------------------------
# Thermal Dynamics
# ---------------------------------------------------------------------------
class ThermalConfig:
    AMBIENT_TEMP_C      = 25.0          # Default ambient temperature (°C)
    DT_MINUTES          = 2.0           # Timestep for temperature update (minutes)
    TEMP_NOISE_STD      = 0.5           # Gaussian noise std dev (°C)
    TEMP_MAX_C          = 150.0         # Equipment temperature upper limit (°C)
    AMBIENT_BUFFER_C    = 5.0           # Min temp = ambient − buffer (°C)
    DIURNAL_AMPLITUDE_C = 8.0           # ±°C diurnal swing around base ambient
    DIURNAL_PEAK_HOUR   = 14            # Hour of day for peak temperature

    # --- Thermal anchor -----------------------------------------------------
    # Equipment temperature at NOMINAL loading (mean incident-line loading = 1.0)
    # measured at the reference ambient AMBIENT_TEMP_C. The per-node heat
    # coefficient is DERIVED from this anchor:
    #
    #     k_i = (RATED_TEMP_C - AMBIENT_TEMP_C) * capacity_i * cooling_i / tau_i
    #
    # so that every node settles at RATED_TEMP_C when its incident lines run at
    # exactly their rating, regardless of its random tau / capacity / cooling
    # draw. Those three then affect only how FAST a node heats and cools (the
    # transient), not where it settles (the equilibrium) — which is precisely
    # what "rated" means physically. Steady-state rise scales with I²:
    #
    #     dT_ss(L) = (RATED_TEMP_C - AMBIENT_TEMP_C) * L²
    #
    # Consequences with the node failure thresholds (105-130°C):
    #   L <= 1.0  -> settles at or below 75°C: heat released >= heat gained,
    #                SAFE INDEFINITELY (no thermal failure is possible)
    #   L ~ 1.26  -> ~105°C, reaches the lowest failure threshold
    #   L ~ 1.45  -> ~130°C, reaches the highest failure threshold
    # so thermal node failure spans L in [1.26, 1.45], lining up with the
    # Phase 1 line tiers (emergency 1.2x, load-dump 1.45x).
    #
    # Previously HEAT_SCALE was a hand-tuned scalar, so the random tau /
    # capacity / cooling draws alone spread the equilibrium temperature at
    # IDENTICAL rated loading over 37.8-101.5°C (2.7x) — one node barely above
    # ambient while another sat just under its failure threshold, with no way
    # to say what temperature "rated load" corresponds to.
    RATED_TEMP_C        = 75.0

    # Global multiplier on the anchored coefficient. 1.0 = exact physical
    # anchor (recommended). Raise to make temperature a more aggressive
    # failure mode, lower to make it more inert.
    HEAT_SCALE          = 1.0


# ---------------------------------------------------------------------------
# Node Properties (topology initialisation)
# ---------------------------------------------------------------------------
class NodeConfig:
    # Node type fractions
    GENERATOR_FRACTION  = 0.22          # ~22 % of nodes are generators
    GENERATOR_MIN_COUNT = 5             # Minimum number of generators
    SUBSTATION_FRACTION = 0.10          # ~10 % of nodes are substations

    # Per-node base load (MW)
    GEN_LOAD_MIN    = 5.0
    GEN_LOAD_MAX    = 20.0
    SUB_LOAD_MIN    = 50.0
    SUB_LOAD_MAX    = 150.0
    LOAD_LOAD_MIN   = 30.0
    LOAD_LOAD_MAX   = 200.0

    # Generation capacity
    TARGET_CAPACITY_FACTOR  = 1.50      # Total capacity = 150 % of total load
    SLACK_CAPACITY_MIN      = 0.30      # Slack bus: 30–40 % of total capacity
    SLACK_CAPACITY_MAX      = 0.40
    DIRICHLET_ALPHA         = 2.0       # Concentration for capacity distribution
    MIN_RESERVE_MARGIN_PCT  = 20        # Alert threshold for reserve margin (%)
    CAPACITY_BOOST_FACTOR   = 1.30      # Upscale if reserve margin is too low

    # Power factor (tan φ = Q/P) by bus type
    PF_GENERATOR_MIN    = 0.97          # Generator bus power factor range
    PF_GENERATOR_MAX    = 0.99
    PF_SUBSTATION_MIN   = 0.92          # Substation bus power factor range
    PF_SUBSTATION_MAX   = 0.96
    PF_LOAD_MIN         = 0.90          # Load bus power factor range
    PF_LOAD_MAX         = 0.97

    # Failure / damage thresholds
    LOADING_FAILURE_MIN     = 1.05
    LOADING_FAILURE_MAX     = 1.15
    LOADING_DAMAGE_OFFSET_MIN = -0.05   # Damage starts below failure threshold
    LOADING_DAMAGE_OFFSET_MAX = -0.10

    VOLTAGE_FAILURE_MIN     = 0.80      # p.u.
    VOLTAGE_FAILURE_MAX     = 0.85
    VOLTAGE_DAMAGE_OFFSET_MIN = 0.04    # Damage starts above failure threshold
    VOLTAGE_DAMAGE_OFFSET_MAX = 0.07

    TEMP_FAILURE_MIN_C      = 105.0
    TEMP_FAILURE_MAX_C      = 130.0
    TEMP_DAMAGE_OFFSET_MIN_C = -15.0
    TEMP_DAMAGE_OFFSET_MAX_C = -25.0

    FREQ_FAILURE_MIN_HZ     = 58.5
    FREQ_FAILURE_MAX_HZ     = 59.2
    FREQ_DAMAGE_OFFSET_MIN  = 0.3
    FREQ_DAMAGE_OFFSET_MAX  = 0.5

    # Equipment age & condition
    EQUIPMENT_AGE_MIN_YR    = 0
    EQUIPMENT_AGE_MAX_YR    = 40
    CONDITION_AGE_COEFF     = 0.008     # Degradation per year: 1 - 0.008*age
    CONDITION_NOISE_STD     = 0.05
    CONDITION_MIN           = 0.6
    CONDITION_MAX           = 1.0

    # Thermal model per node
    THERMAL_TAU_MIN     = 10.0          # Thermal time constant (minutes)
    THERMAL_TAU_MAX     = 30.0
    THERMAL_CAP_MIN     = 0.8           # Thermal capacity (normalised)
    THERMAL_CAP_MAX     = 1.2
    COOLING_EFF_MIN     = 0.7           # Cooling effectiveness
    COOLING_EFF_MAX     = 1.0


# ---------------------------------------------------------------------------
# Grid Topology Generation
# ---------------------------------------------------------------------------
class TopologyConfig:
    NUM_ZONES               = 4
    ZONE_CENTERS            = [(-50, -50), (50, -50), (-50, 50), (50, 50)]  # (x, y) km
    ZONE_SPREAD_STD         = 20.0      # Node position std dev within zone (km)
    INTRA_ZONE_CONN_MIN     = 2         # Min connections per node within zone
    INTRA_ZONE_CONN_MAX     = 5         # Max connections per node within zone (exclusive)
    TIE_LINES_MIN           = 2         # Min tie lines between adjacent zones
    TIE_LINES_MAX           = 4         # Max tie lines between adjacent zones (exclusive)
    DEFAULT_NUM_NODES       = 118       # Default grid size (IEEE 118-bus reference)


# ---------------------------------------------------------------------------
# Cascade Propagation
# ---------------------------------------------------------------------------
class CascadeConfig:
    STRESS_TO_LOADING_FACTOR    = 0.40  # Loading increase per unit stress
    STRESS_TO_VOLTAGE_FACTOR    = 0.15  # Voltage decrease per unit stress
    STRESS_TO_TEMP_FACTOR       = 25.0  # Temperature increase (°C) per unit stress
    FAILURE_DELAY_MIN           = 0.1   # Physical propagation delay (minutes)
    FAILURE_DELAY_MAX           = 0.5
    STRESS_DECAY                = 0.80  # Downstream stress decay factor

    # Independent line tripping (Phase 0 — MISO data gap plan)
    # [MISO RTO-SPEC-006: "Switching-device (breaker/switch) open/closed
    #  status" — prerequisite: gives breakers line-level trip events to report]
    # A line trips on its own when apparent power exceeds a rating tier for a
    # sustained number of consecutive timesteps.
    # LINE_TRIP_LOADING_RATIO is the scalar fallback used when no per-edge
    # emergency ratings are supplied (pre-Phase-1 behavior).
    LINE_TRIP_LOADING_RATIO     = 1.2
    LINE_TRIP_SUSTAIN_STEPS     = 3   # steps above emergency rating before trip
    LINE_TRIP_LOAD_DUMP_STEPS   = 1   # steps above load-dump rating before trip

    # Multi-rating lines (Phase 1 — MISO data gap plan)
    # [MISO RTO-SPEC-006: "Line ratings — normal / emergency (LEP) / load-dump
    #  set" (was: single thermal limit)]
    # Per-edge rating tiers as random multiples of the normal rating
    # (thermal_limits). Emergency = MISO's LEP rating; load-dump is the
    # minutes-only ceiling. loading_ratios stay normalized by the NORMAL
    # rating everywhere — tiers are thresholds on that ratio, so heat, node
    # thresholds, and training features keep their calibration.
    EMERGENCY_RATING_FACTOR_MIN = 1.15
    EMERGENCY_RATING_FACTOR_MAX = 1.30
    LOAD_DUMP_RATING_FACTOR_MIN = 1.40
    LOAD_DUMP_RATING_FACTOR_MAX = 1.55

    # Cascade adjacency propagation weights
    WEIGHT_GEN_TO_SUB   = 0.9
    WEIGHT_SUB_TO_LOAD  = 0.7
    WEIGHT_GEN_TO_LOAD  = 0.7
    WEIGHT_LOAD_TO_LOAD = 0.5
    WEIGHT_DEFAULT      = 0.6


# ---------------------------------------------------------------------------
# Branch Typing & Tap Devices (Phase 2 — MISO data gap plan)
# [MISO RTO-SPEC-006: "Transformer flows + LTC tap positions" and
#  "Phase shifter (PAR) flows + tap positions"]
# ---------------------------------------------------------------------------
class BranchConfig:
    # Branch type codes (per directed edge, mirrored across each pair)
    TYPE_LINE        = 0
    TYPE_TRANSFORMER = 1     # LTC transformer — regulates voltage magnitude
    TYPE_PAR         = 2     # phase-angle regulator — reroutes active power

    # Fractions of PHYSICAL (undirected) branches assigned each device type
    TRANSFORMER_FRACTION = 0.12
    PAR_FRACTION         = 0.03

    # LTC (load tap changer): tap ∈ [MIN, MAX], each step = STEP_PU voltage
    # ratio. The controller steps ±1 per timestep (mechanical delay) toward
    # V_SETPOINT on the regulated bus, ignoring a ±DEADBAND band.
    LTC_TAP_MIN        = -16
    LTC_TAP_MAX        = 16
    LTC_STEP_PU        = 0.00625     # 0.625 % per step (standard 33-step LTC)
    LTC_V_SETPOINT_PU  = 1.0
    # Deadband must COVER the AVR setpoint spread (AVR_SETPOINT 0.99-1.03,
    # PV_V_SET 0.98-1.02), otherwise the LTC and the generator voltage
    # schedules fight each other: any bus held above the band is a permanent
    # out-of-band error the tap keeps chasing. At the old +-0.01 the band was
    # [0.99, 1.01] while AVR ran to 1.03, so transformers ratcheted one step
    # per timestep until they clipped, rerouting flow and tripping lines at
    # constant load. +-0.03 spans the full schedule range.
    LTC_DEADBAND_PU    = 0.03
    LTC_DEAD_BUS_V_PU  = 0.5         # regulated bus below this = de-energized; hold tap

    # Anti-windup: a tap that has moved but produces less than
    # MIN_V_RESPONSE_PU of change at its regulated bus for STALL_STEPS
    # consecutive steps has no practical authority (e.g. the bus is pinned by
    # a generator, or the transformer is electrically weak relative to the
    # rest of the network). Block it rather than let it ratchet to the limit.
    LTC_MIN_V_RESPONSE_PU = 0.0005
    LTC_STALL_STEPS       = 3

    # PAR (phase-angle regulator): angle ∈ [MIN, MAX] degrees, STEP per
    # timestep. Controller relieves its OWN loading: above LOADING_TARGET it
    # steps the shift against the flow direction (DC approx: P ≈ (Δθ + α)/x).
    PAR_ANGLE_MIN_DEG  = -30.0
    PAR_ANGLE_MAX_DEG  = 30.0
    PAR_STEP_DEG       = 2.0
    PAR_LOADING_TARGET = 0.9


# ---------------------------------------------------------------------------
# Reactive Support & Voltage Regulation (Phase 3 — MISO data gap plan)
# [MISO RTO-SPEC-006: "Voltage-regulation equipment (AVR, capacitor/reactor,
#  SVC) status"]
# ---------------------------------------------------------------------------
class ShuntConfig:
    # Switchable capacitor banks (inject Q, prop voltage up under heavy load)
    CAP_BANK_FRACTION    = 0.25    # fraction of LOAD buses with a cap bank
    CAP_BANK_MVAR_MIN    = 20.0
    CAP_BANK_MVAR_MAX    = 60.0
    CAP_IN_V_PU          = 0.98    # switch IN below this (hysteresis pair)
    CAP_OUT_V_PU         = 1.02    # switch OUT above this

    # Switchable reactor banks (absorb Q, pull voltage down at light load)
    REACTOR_FRACTION     = 0.10    # fraction of SUBSTATION buses with a reactor
    REACTOR_MVAR_MIN     = 20.0
    REACTOR_MVAR_MAX     = 50.0
    REACTOR_IN_V_PU      = 1.02
    REACTOR_OUT_V_PU     = 0.99

    # SVCs — continuous, fast reactive support: Q = clip(GAIN·(Vset−V), ±Qmax)
    SVC_FRACTION         = 0.05    # fraction of ALL buses with an SVC
    SVC_QMAX_MVAR_MIN    = 30.0
    SVC_QMAX_MVAR_MAX    = 80.0
    SVC_GAIN_MVAR_PER_PU = 800.0
    SVC_V_SETPOINT_PU    = 1.0

    # AVR setpoints (generator terminal-voltage targets, telemetered)
    AVR_SETPOINT_MIN_PU  = 0.99
    AVR_SETPOINT_MAX_PU  = 1.03

    DEAD_BUS_V_PU        = 0.5     # de-energized bus: devices disconnect/hold


# ---------------------------------------------------------------------------
# Remedial Action Schemes (Phase 4 — MISO data gap plan)
# [MISO RTO-SPEC-006: "RAS arming status (Armed/Available/Triggered/Operated)"]
# ---------------------------------------------------------------------------
class RASConfig:
    N_SCHEMES        = 6      # RAS schemes per grid

    # Arming/trigger thresholds on the monitored line's loading ratio.
    # Trigger sits BELOW the Phase-0/1 emergency trip band (sustained ≥1.15×):
    # a RAS exists precisely to act before the line protection does.
    ARM_LOADING      = 0.80   # scheme arms above this
    TRIGGER_LOADING  = 1.05   # scheme fires above this (single-shot)

    # Arming status codes (telemetered; MISO vocabulary)
    STATUS_AVAILABLE = 0      # in service, conditions normal
    STATUS_ARMED     = 1      # arming condition met — will fire on trigger
    STATUS_TRIGGERED = 2      # trigger detected this timestep (action firing)
    STATUS_OPERATED  = 3      # action executed — latched for the scenario

    # Action types (chosen per scheme at assignment)
    ACTION_TRIP_GEN  = 0      # drop a generator's injection to 0 (controlled)
    ACTION_SHED_LOAD = 1      # shed a fraction of load at a bus
    ACTION_OPEN_LINE = 2      # open a designated relief line

    SHED_FRACTION    = 0.30   # load fraction shed by ACTION_SHED_LOAD

    RAS_FEATURES     = 3      # per-scheme stream: [status, monitored_loading, action_type]


# ---------------------------------------------------------------------------
# Inter-area exchange: tie lines & interchange (Phase 5 — MISO data gap plan)
# [MISO RTO-SPEC-006: "Tie-line flows & net actual interchange",
#  "Dynamic schedules (interchange MW)"]
# ---------------------------------------------------------------------------
class InterchangeConfig:
    # Tie lines connect our footprint to a synthetic neighbouring RC. They are
    # chosen as the branches farthest from the network centroid — real ties sit
    # at the boundary of a footprint, not in its interior.
    N_TIE_LINES        = 8        # physical branches designated as ties

    # Scheduled interchange (MW, positive = we EXPORT to the neighbour). The
    # schedule is a slow block profile: it holds for a dwell of timesteps and
    # then re-steps, mimicking hourly interchange blocks rather than smooth
    # drift. ACE = schedule - actual, so a schedule step is a genuine
    # disturbance the model can learn to anticipate.
    SCHEDULE_BASE_MW   = 0.0      # centre of the schedule distribution
    SCHEDULE_SPAN_MW   = 400.0    # schedule drawn from +-SPAN around BASE
    SCHEDULE_DWELL_MIN = 8        # timesteps a schedule block holds (min)
    SCHEDULE_DWELL_MAX = 16       # ... and max, re-drawn each block

    # Normalisation divisor for the broadcast SCADA columns (MW -> ~unit scale)
    INTERCHANGE_SCALE_MW = 1000.0


# ---------------------------------------------------------------------------
# Scheduled outages & de-rates (Phase 6 — MISO data gap plan)
# [MISO RTO-SPEC-006: "Transmission/generation outage schedules + de-rates"]
# ---------------------------------------------------------------------------
class OutageConfig:
    # MASTER SWITCH. When False, scenario generation injects NO planned
    # outages or scheduled de-rates: branch_status stays binary (1 closed /
    # 0 forced-open, never -1), no line/generator is taken out on a schedule,
    # and rating_derate stays at its static baseline. The Phase 6 machinery
    # (outages.py, this config, the simulator wiring, the tests) remains fully
    # in place and re-enables with this one flag.
    #
    # Rationale for leaving it OFF: a planned outage is a human-scheduled
    # event, not grid FAILURE behaviour, so it is noise for the core
    # cascade-prediction task. Turn it on to train the planned-vs-forced
    # distinction (the ternary branch_status) once that is wanted.
    ENABLED = False

    # branch_status is TERNARY from Phase 6 onward. A planned outage and a
    # forced trip both open the branch, but they mean opposite things: one is
    # an operator decision known in advance, the other is the failure the
    # model exists to predict. Encoding both in one column keeps the feature
    # width unchanged and keeps the two facts in the same place.
    #
    # ANY consumer that masks on branch_status MUST test `== STATUS_CLOSED`,
    # never truthiness — STATUS_PLANNED_OPEN is -1.0, which is truthy, so a
    # planned-open branch would silently stay in the GNN adjacency.
    STATUS_CLOSED       = 1.0
    STATUS_FORCED_OPEN  = 0.0
    STATUS_PLANNED_OPEN = -1.0

    # --- Scheduled TRANSMISSION outages ---------------------------------
    LINE_OUTAGE_PROBABILITY  = 0.35   # scenarios that carry any line outage
    N_LINE_OUTAGES_MIN       = 1
    N_LINE_OUTAGES_MAX       = 3
    LINE_OUTAGE_DUR_MIN      = 6      # timesteps out of service
    LINE_OUTAGE_DUR_MAX      = 20

    # --- Scheduled GENERATOR outages ------------------------------------
    # The unit is derated to zero output but its BUS stays energised — a
    # planned outage is not a node failure and must never be labelled as one.
    GEN_OUTAGE_PROBABILITY   = 0.30
    N_GEN_OUTAGES_MIN        = 1
    N_GEN_OUTAGES_MAX        = 2
    GEN_OUTAGE_DUR_MIN       = 8
    GEN_OUTAGE_DUR_MAX       = 25

    # Outages start early enough in the sequence to be visible before any
    # cascade, so the model can learn the planned-vs-forced distinction.
    START_FRACTION_MIN       = 0.05
    START_FRACTION_MAX       = 0.60

    # OUTAGE SCREENING. A real planned outage is studied first and rejected
    # if removing that branch creates a thermal violation. Without a screen,
    # a uniformly-random outage lands on a backbone corridor and its flow
    # dumps onto a floor-rated parallel path — measured a jump from 1.14x to
    # 6.85x rating in ONE timestep, which then took the whole grid down from
    # a PLANNED event. Ratings were calibrated from base-case flow, so the
    # rating is a proxy for normal loading: exclude the busiest corridors
    # from outage candidacy.
    EXCLUDE_TOP_RATED_FRAC   = 0.40   # top 40% by rating are never outaged

    # --- Scheduled DE-RATE events ---------------------------------------
    DERATE_PROBABILITY       = 0.40
    N_DERATE_EVENTS_MIN      = 1
    N_DERATE_EVENTS_MAX      = 4
    DERATE_FACTOR_MIN        = 0.70   # rating multiplier while in force
    DERATE_FACTOR_MAX        = 0.92
    DERATE_DUR_MIN           = 8
    DERATE_DUR_MAX           = 30

    # --- AMBIENT-driven de-rate -----------------------------------------
    # A conductor's current rating falls as ambient rises: the thermal limit
    # is defined at a reference ambient, and a hot afternoon eats headroom.
    # ~0.4 %/°C is a common planning rule of thumb. This couples to the
    # diurnal ambient the simulator already swings (ThermalConfig).
    AMBIENT_REF_C            = 25.0   # nameplate rating holds at/below this
    AMBIENT_DERATE_PER_C     = 0.004  # rating lost per °C above the reference
    AMBIENT_DERATE_FLOOR     = 0.80   # ambient alone never de-rates past this

    # Combined de-rate is clamped here so a scheduled event stacked on a hot
    # day cannot drive a rating to something unphysical.
    TOTAL_DERATE_FLOOR       = 0.50


# ---------------------------------------------------------------------------
# Storage / ESR (Phase 5 — MISO data gap plan)
# [MISO RTO-SPEC-006: "Energy storage resources: SOC, MW, MVAr"]
# ---------------------------------------------------------------------------
class StorageConfig:
    # Storage is a SEPARATE per-node attribute rather than a new node_types
    # value: node_types == 1 (generator) is load-bearing in the power flow,
    # the LTC regulated-bus choice, RAS targeting and dispatch, so overloading
    # it would silently change all of those. A bus is a storage bus iff
    # storage_power_mw > 0.
    N_STORAGE_NODES    = 6        # buses fitted with an ESR

    POWER_MW_MIN       = 20.0     # charge/discharge power rating
    POWER_MW_MAX       = 80.0
    DURATION_H_MIN     = 2.0      # energy capacity = power x duration
    DURATION_H_MAX     = 4.0
    SOC_INIT_FRACTION  = 0.55     # starting state of charge

    # Frequency droop: discharge to support LOW frequency, charge on HIGH.
    #   flow_mw = clip(DROOP_MW_PER_HZ * (f_nom - f), -power, +power)
    # positive flow = DISCHARGING (injecting MW, depleting SOC).
    #
    # STABILITY — read before retuning. Storage MW feeds the generation sum
    # that FrequencyDynamicsSimulator uses, so the droop closes a feedback
    # loop around the frequency model. That model's sensitivity is
    #     df/dP = BASE_FREQUENCY / (D_TOTAL * total_load)
    #           = 60 / (0.04 * 8500) ~= 0.18 Hz per MW
    # so the loop gain is  0.18 * N_STORAGE_NODES * DROOP_MW_PER_HZ.
    # At the original 120 MW/Hz that gain was ~127: a tiny frequency dip
    # produced a storage response that overshot by two orders of magnitude,
    # flipped sign, diverged within one step and collapsed the whole grid on
    # underfrequency at t=1. MAX_LOOP_GAIN below bounds it structurally in
    # StorageController.update(), so the cap holds even if this gain is
    # raised or the grid is resized.
    # Closed-loop pole is |1 - a - a*G| with a = 1-exp(-dt/TAU_GOV) ~= 0.865
    # and G = 0.18 * N_STORAGE_NODES * DROOP_MW_PER_HZ. Measured:
    #   droop 120 -> G=127  pole=110   diverges (the t=1 collapse)
    #   droop   5 -> G=5.3  pole=4.4   diverges
    #   droop   1 -> G=1.06 pole=0.78  stable but ~oscillatory
    #   droop 0.2 -> G=0.21 pole=0.05  settles in ~3 steps, no limit cycle
    # 0.2 also keeps the response PROPORTIONAL (never hits MAX_LOOP_GAIN in
    # the normal band), which matters: a saturating cap turns the droop into
    # a bang-bang controller and leaves a sustained +-0.19 Hz limit cycle.
    #
    # CAVEAT — this gain is far below a real ESR's frequency response. It is
    # sized to THIS frequency model, whose sensitivity (0.18 Hz per MW of
    # imbalance, from D_TOTAL = 0.04) is roughly two orders of magnitude
    # twitchier than a real interconnection. The consequence is that storage
    # MW/SOC barely move, so those SCADA columns carry little signal. Raising
    # the response meaningfully requires recalibrating FrequencyConfig.D_TOTAL
    # first — a change to PRE-EXISTING physics that would alter every
    # scenario's frequency dynamics and needs its own validation.
    DROOP_MW_PER_HZ    = 0.2
    MAX_LOOP_GAIN      = 0.25     # structural backstop; scales aggregate response
    DEADBAND_HZ        = 0.02     # no response inside +-this of nominal

    # Reactive capability, as a fraction of the MW rating (inverter-coupled
    # ESRs can supply Q largely independent of P).
    QMAX_FRACTION      = 0.33

    # Round-trip efficiency applied on the CHARGE direction only.
    CHARGE_EFFICIENCY  = 0.92


# ---------------------------------------------------------------------------
# Time-Series Simulation
# ---------------------------------------------------------------------------
class SimulationConfig:
    DEFAULT_SEQUENCE_LENGTH = 30        # Timesteps per scenario
    RAMP_FRACTION_MIN       = 0.65      # Minimum Fraction of sequence used for stress ramp
    RAMP_FRACTION_MAX       = 0.85      # Maximum Fraction of sequence used for stress ramp

    GENERATION_MARGIN       = 1.02      # Generation oversizing factor vs total load

    # Load noise (std dev as fraction of load)
    LOAD_NOISE_HIGH_STRESS  = 0.05
    LOAD_NOISE_LOW_STRESS   = 0.02
    LOAD_NOISE_CLIP_FACTOR  = 1.0       # Clip at ±N × std dev

    # Load multiplier formula: base + current_stress * slope
    LOAD_MULT_HIGH_BASE     = 0.7       # Base multiplier for high-stress scenarios
    LOAD_MULT_HIGH_SLOPE    = 0.4
    LOAD_MULT_LOW_BASE      = 0.5       # Base multiplier for low-stress scenarios
    LOAD_MULT_LOW_SLOPE     = 0.4

    # Grid collapse reporting threshold
    COLLAPSE_FAILURE_RATIO  = 0.9       # Fraction of failed nodes = collapse

    # Physics cascade propagation cap: max additional failures per cascade wave
    # (expressed as a fraction of total nodes, on top of the initial trigger failures)
    CASCADE_MAX_SPREAD_FRACTION = 0.30

    # Ambient temperature model
    AMBIENT_BASE_MIN_C      = 25.0
    AMBIENT_BASE_MAX_C      = 35.0      # base = 25 + 10 * rand()


# ---------------------------------------------------------------------------
# Scenario Orchestration
# ---------------------------------------------------------------------------
class ScenarioConfig:
    DEFAULT_NUM_NODES       = TopologyConfig.DEFAULT_NUM_NODES
    DEFAULT_SEQUENCE_LENGTH = SimulationConfig.DEFAULT_SEQUENCE_LENGTH
    DEFAULT_NUM_NORMAL      = 100
    DEFAULT_NUM_CASCADE     = 80
    DEFAULT_NUM_STRESSED    = 20
    DEFAULT_BATCH_SIZE      = 10
    DEFAULT_SEED            = 42
    MAX_RETRIES             = 10

    # Stress level ranges by scenario type
    # Stress level ranges by scenario type — contiguous, non-overlapping, covering [0, 1].
    # Normal:   0.00–0.55  (safe operation, no failures expected)
    # Stressed: 0.55–0.72  (near-miss region — high load, no cascade)
    # Cascade:  0.72–1.00  (critical stress — failures propagate)
    CASCADE_STRESS_MIN      = 0.72
    CASCADE_STRESS_MAX      = 1.00
    STRESSED_STRESS_MIN     = 0.55
    STRESSED_STRESS_MAX     = 0.72
    NORMAL_STRESS_MIN       = 0.00
    NORMAL_STRESS_MAX       = 0.55


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetConfig:
    BASE_MVA                = PowerSystemConfig.SN_MVA
    BASE_FREQUENCY          = PowerSystemConfig.BASE_FREQUENCY
    DEFAULT_NUM_NODES       = TopologyConfig.DEFAULT_NUM_NODES

    # Default path where the pre-generated grid topology pickle is stored.
    # Callers that do not supply an explicit topology_file will fall back to
    # this value so that the same grid structure is reused across runs.
    DEFAULT_TOPOLOGY_FILE   = "data/grid_topology.pkl"

    TRAIN_RATIO             = 0.70
    VAL_RATIO               = 0.15
    TEST_RATIO              = 0.15
    RATIO_TOLERANCE         = 1e-6      # Floating-point tolerance for ratio validation

    CASCADE_LABEL_THRESHOLD = 0.5       # Threshold to classify node as cascade
    AUGMENTATION_NOISE_STD  = 0.01      # Gaussian noise for training augmentation


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
class TrainingConfig:
    LEARNING_RATE           = 0.0001    # Adam initial learning rate
    GRAD_CLIP               = 20.0      # Default CLI gradient clipping max norm
    TRAINER_MAX_GRAD_NORM   = 5.0       # Trainer class default max grad norm
    EPOCHS                  = 100       # Default number of training epochs
    BATCH_SIZE              = 8         # Default batch size
    PATIENCE                = 10        # Early-stopping patience (epochs)
    WEIGHT_DECAY            = 1e-3      # Adam weight decay (L2 regularisation)
    SCHEDULER_PATIENCE      = 5         # ReduceLROnPlateau patience
    CASCADE_THRESHOLD       = 0.25      # Decision threshold for cascade prediction
    NODE_THRESHOLD          = 0.25      # Decision threshold for node failure
    # IMPROVED: FBETA changed from 0.5 → 1.0 so the validation threshold search
    # directly maximises standard F1 (the metric we want to report) rather than
    # F-beta(0.5) which is precision-biased and consistently under-reports Node F1.
    FBETA                   = 1.0       # Beta for F-beta score (1.0 = standard F1)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class ModelConfig:
    EMBEDDING_DIM       = 128           # Embedding vector dimension
    HIDDEN_DIM          = 128           # Hidden representation dimension
    NUM_GNN_LAYERS      = 3             # Number of stacked GNN layers
    HEADS               = 4             # Multi-head attention heads
    DROPOUT             = 0.3           # Default model dropout rate
    DROPOUT_TRAIN       = 0.5           # Default CLI training dropout rate
    HEAD_DROPOUT_HIGH   = 0.4           # Dropout for failure_prob / freq / risk / timing heads
    HEAD_DROPOUT_LOW    = 0.3           # Dropout for voltage / angle / flow / temperature heads
    LSTM_NUM_LAYERS     = 3             # LSTM layer count in TemporalGNNCell
    LSTM_DROPOUT        = 0.3           # LSTM internal dropout
    # Edge feature vector length. Phase 1 (MISO plan) appended 3 cols to the
    # original 7: [7] branch_status (binary), [8] emergency limit, [9] load-dump
    # limit. Existing indices 0–6 unchanged (append-only invariant). Width
    # change requires model retrain.
    # [MISO RTO-SPEC-006: col 7 → "Switching-device open/closed status";
    #  cols 8–9 → "Line ratings — normal/emergency(LEP)/load-dump set"]
    # Phase 2 appended 3 more: [10] branch_type (0=line/1=LTC/2=PAR),
    # [11] tap_position (LTC steps), [12] tap_angle (PAR degrees).
    # [MISO RTO-SPEC-006: "Transformer flows + LTC tap positions",
    #  "Phase shifter (PAR) flows + tap positions"]
    # Phase 5 appended 1 more: [13] is_tie_line (binary, raw — no normalisation).
    # [MISO RTO-SPEC-006: "Tie-line flows & net actual interchange"]
    EDGE_FEATURES       = 14
    RISK_DIM            = 7             # Output dimension of risk assessment head
    GAT_DROPOUT         = 0.1           # Graph attention dropout
    LEAKY_RELU_SLOPE    = 0.2           # LeakyReLU negative slope in GAT


# ---------------------------------------------------------------------------
# Physics Supervision (next-step prediction from LSTM hidden state h_T)
# ---------------------------------------------------------------------------
class PhysicsPredConfig:
    LAMBDA_VOLTAGE  = 0.5   # MSE loss: predicted V_{T} vs scada[:, -1, :, 0]
    LAMBDA_TEMP     = 0.5   # MSE loss: predicted temp_{T} vs scada[:, -1, :, 5]
    LAMBDA_FLOW     = 0.3   # MSE loss: predicted P_ij_{T} vs edge_attr[:, -1, :, 5]


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
class LossConfig:
    # Default lambda weights (PhysicsInformedLoss constructor defaults)
    # IMPROVED: LAMBDA_PREDICTION raised from 10 → 30 to give the node-failure
    # detection task a larger share of the gradient budget vs physics auxiliaries
    # (reactive, timing, temperature, etc.).  In the old config the physics losses
    # collectively contributed ~95% of the gradient — the prediction task was
    # starved.  With 30 the prediction focal loss now dominates appropriately.
    LAMBDA_PREDICTION   = 30.0
    LAMBDA_POWERFLOW    = 0.1
    LAMBDA_RISK         = 0.1
    # IMPROVED: timing lambda reduced from 8.0 → 2.0.
    # With the new cascade_obs_steps=7 truncation the model sees early cascade
    # failures; timing becomes easier to learn.  Reducing its weight frees
    # gradient budget for the prediction task.
    LAMBDA_TIMING       = 2.0
    LAMBDA_ACTIVE_FLOW  = 0.1
    # TEMPERATURE: loss is MSE(pred/100, true/100) = MSE_raw/10000.  With raw MSE ≈ 50
    # (7°C mean error), effective contribution = 0.05 * 50/10000 ≈ 0.00025 — negligible.
    # Raising to 10.0 yields 10.0 * 50/10000 = 0.05, on par with powerflow/reactive.
    LAMBDA_TEMPERATURE  = 10.0
    # FREQUENCY: now supervised directly against SCADA ground-truth (fixed from physics
    # formula with wrong POWER_TO_FREQ constant).  Small upward nudge from 0.08 → 0.10
    # since the supervision signal is now correct and informative.
    LAMBDA_FREQUENCY    = 0.1
    LAMBDA_REACTIVE     = 0.1
    # VOLTAGE: down-weighted from 1.0 → 0.3.  At 1.0 it consumed ~10% of the total
    # gradient budget for an auxiliary task, crowding out failure prediction and timing.
    LAMBDA_VOLTAGE      = 0.3
    LAMBDA_CAPACITY     = 0.05
    LAMBDA_PARENT       = 0.3   # Weight for causal parent prediction loss (reduced from 0.5)
    PARENT_NON_TRIGGER_WEIGHT = 5.0  # Cross-entropy class weight for actual-parent labels (vs trigger=1.0)

    # Focal loss parameters
    FOCAL_ALPHA         = 0.85          # PhysicsInformedLoss default
    FOCAL_GAMMA         = 2.0
    # IMPROVED: FOCAL_ALPHA_TRAIN raised from 0.25 → 0.75.
    # With 0.25 the focal loss penalised the negative class (non-failing nodes)
    # 3× more than the positive class, despite the positive class being the
    # minority (~17-20% of nodes).  alpha=0.75 correctly up-weights the minority
    # positive class, improving recall and the joint F1 optimum.
    FOCAL_ALPHA_TRAIN   = 0.75          # train_model.py (calibrated path)
    FOCAL_ALPHA_FALLBACK = 0.65         # train_model.py (uncalibrated fallback)

    # Dynamic calibration
    CALIB_NUM_BATCHES       = 20        # Batches used for loss-weight calibration
    CALIB_LAMBDA_PREDICTION = 50.0      # Prediction weight used during calibration
    CALIB_MIN_MAGNITUDE     = 1e-9      # Floor for target_magnitude to avoid div-by-zero

    # Physics scaling constants
    TEMPERATURE_SCALE   = 100.0         # Temperature normalisation divisor (deg C)
    POWER_TO_FREQ       = 10.0          # Power-imbalance to frequency-deviation factor


# ---------------------------------------------------------------------------
# Embedding Networks
# ---------------------------------------------------------------------------
class EmbeddingConfig:
    # -- Environmental -------------------------------------------------------
    ENV_SATELLITE_CHANNELS  = 12        # Satellite image input channels
    ENV_WEATHER_FEATURES    = 80        # Weather sequence feature width
    ENV_THREAT_FEATURES     = 6         # Threat indicator vector length
    ENV_SAT_HIDDEN          = 32        # Satellite CNN output channels
    ENV_WEATHER_HIDDEN      = 32        # Weather LSTM hidden size
    ENV_THREAT_HIDDEN       = 32        # Threat encoder hidden size
    ENV_WEATHER_LSTM_LAYERS = 2         # Weather LSTM layer count

    # -- Infrastructure ------------------------------------------------------
    # SCADA width: 18 base + 4 Phase 3 reactive-support columns
    # ([18] cap bank status, [19] reactor status, [20] SVC Q output,
    #  [21] AVR setpoint). [MISO RTO-SPEC-006: voltage-regulation equipment]
    # Phase 5 appended 6:
    #   PER-NODE (zero on non-storage buses):
    #     [22] storage SOC fraction   (soc_mwh / capacity_mwh, 0-1)
    #     [23] storage MW  (+ discharge / - charge, / power rating)
    #     [24] storage MVAr           (/ Qmax)
    #   BROADCAST system-wide scalars — identical on every node, exactly as
    #   current_frequency already is in col 6. Not local quantities; see the
    #   plan's note. All divided by InterchangeConfig.INTERCHANGE_SCALE_MW:
    #     [25] net actual interchange (sum of signed tie flows)
    #     [26] scheduled interchange
    #     [27] ACE = schedule - actual
    # [MISO RTO-SPEC-006: tie-line flows / net actual interchange, dynamic
    #  schedules, energy storage resources]
    INFRA_SCADA_FEATURES    = 28        # SCADA measurement feature width
    INFRA_PMU_FEATURES      = 8         # PMU measurement feature width
    INFRA_EQUIPMENT_FEATURES = 10       # Equipment status feature width
    INFRA_SCADA_HIDDEN      = 64        # SCADA encoder hidden size
    INFRA_PMU_HIDDEN        = 32        # PMU projection hidden size
    INFRA_EQUIP_HIDDEN      = 32        # Equipment encoder hidden size

    # -- Robotic -------------------------------------------------------------
    ROBOT_VISUAL_CHANNELS   = 3         # Visual camera input channels (RGB)
    ROBOT_THERMAL_CHANNELS  = 1         # Thermal camera input channels
    ROBOT_SENSOR_FEATURES   = 12        # Robotic sensor vector length
    ROBOT_VIS_HIDDEN        = 32        # Visual CNN output channels
    ROBOT_THERM_HIDDEN      = 16        # Thermal CNN output channels
    ROBOT_SENSOR_HIDDEN     = 32        # Sensor encoder hidden size
    ROBOT_FUSION_INPUT      = 80        # Fusion input = VIS + THERM + SENSOR

    # -- Shared --------------------------------------------------------------
    DROPOUT_CNN             = 0.2       # Spatial (2-D) dropout in CNN blocks
    DROPOUT_FC              = 0.3       # Dropout in fully-connected blocks

    # -- Node-level feature MLP ------------------------------------------
    # Feature layout per node per timestep (136 total; Phase 3 widened SCADA
    # 18→22, shifting every later offset):
    #   [0:22]   SCADA measurements          (22; 18-21 reactive support)
    #   [22:30]  PMU measurements             ( 8)
    #   [30:40]  Equipment status            (10)
    #   [40:41]  Active power injection       ( 1)
    #   [41:42]  Reactive power injection     ( 1)
    #   [42:84]  1-step temporal deltas of [0:42]  (42)
    #   [84:126] 2-step temporal deltas of [0:42]  (42)
    #   [126]    Normalised timestep position ( 1)
    #   [127]    TTF voltage  — normalised steps to voltage threshold  ( 1)
    #   [128]    TTF temp     — normalised steps to temperature threshold ( 1)
    #   [129]    TTF freq     — normalised steps to frequency threshold  ( 1)
    #   [130]    TTF loading  — normalised steps to loading threshold    ( 1)
    #   [131:136] Cascade susceptibility      ( 5)
    #
    # The 4 new TTF (time-to-failure) features encode physics-based estimates
    # of how many steps remain before each failure condition is breached.
    # They directly provide the timing signal needed by the TimingHead.
    NODE_FEATURE_BASE_DIM   = 48        # SCADA(28)+PMU(8)+equip(10)+inj(2) before deltas (P3: 38→42, P5: 42→48)
    NODE_FEATURE_DIM        = 154       # 48*3 + 1 t_pos + 4 TTF + 5 susceptibility (P3: 124→136, P5: 136→154)
    NODE_MLP_HIDDEN_1       = 256       # First hidden layer of NodeFeatureMLP
    NODE_MLP_HIDDEN_2       = 128       # Second hidden layer of NodeFeatureMLP


# ---------------------------------------------------------------------------
# Single entry-point: import only Settings everywhere
# ---------------------------------------------------------------------------
class Settings:
    """Aggregates all config sub-classes under one name.

    Usage::
        from .config import Settings
        Settings.PowerSystem.SN_MVA
        Settings.Frequency.BASE_FREQUENCY
    """
    PowerSystem   = PowerSystemConfig
    PowerFlow     = PowerFlowConfig
    LineImpedance = LineImpedanceConfig
    UFLS          = UFLSConfig
    Frequency     = FrequencyConfig
    Thermal       = ThermalConfig
    Node          = NodeConfig
    Topology      = TopologyConfig
    Cascade       = CascadeConfig
    Branch        = BranchConfig
    Shunt         = ShuntConfig
    RAS           = RASConfig
    Interchange   = InterchangeConfig
    Storage       = StorageConfig
    Outage        = OutageConfig
    Simulation    = SimulationConfig
    Scenario      = ScenarioConfig
    Dataset       = DatasetConfig
    Training      = TrainingConfig
    Model         = ModelConfig
    Loss          = LossConfig
    Embedding     = EmbeddingConfig
    PhysicsPred   = PhysicsPredConfig
