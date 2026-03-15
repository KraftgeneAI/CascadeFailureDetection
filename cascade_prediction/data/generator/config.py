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
    SN_MVA        = 100.0    # System base MVA (per-unit base)
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

    # Cascade adjacency propagation weights
    WEIGHT_GEN_TO_SUB   = 0.9
    WEIGHT_SUB_TO_LOAD  = 0.7
    WEIGHT_GEN_TO_LOAD  = 0.7
    WEIGHT_LOAD_TO_LOAD = 0.5
    WEIGHT_DEFAULT      = 0.6


# ---------------------------------------------------------------------------
# Time-Series Simulation
# ---------------------------------------------------------------------------
class SimulationConfig:
    DEFAULT_SEQUENCE_LENGTH = 30        # Timesteps per scenario
    RAMP_FRACTION           = 0.67      # Fraction of sequence used for stress ramp
    GENERATION_MARGIN       = 1.02      # Generation oversizing factor vs total load

    # Load noise (std dev as fraction of load)
    LOAD_NOISE_HIGH_STRESS  = 0.05
    LOAD_NOISE_LOW_STRESS   = 0.02
    LOAD_NOISE_CLIP_FACTOR  = 2.0       # Clip at ±N × std dev

    # Load multiplier formula: base + current_stress * slope
    LOAD_MULT_HIGH_BASE     = 0.7       # Base multiplier for high-stress scenarios
    LOAD_MULT_HIGH_SLOPE    = 0.4
    LOAD_MULT_LOW_BASE      = 0.5       # Base multiplier for low-stress scenarios
    LOAD_MULT_LOW_SLOPE     = 0.4

    # Grid collapse reporting threshold
    COLLAPSE_FAILURE_RATIO  = 0.9       # Fraction of failed nodes = collapse

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
    CASCADE_STRESS_MIN      = 0.70
    CASCADE_STRESS_MAX      = 1.00
    STRESSED_STRESS_MIN     = 0.50
    STRESSED_STRESS_MAX     = 0.62
    NORMAL_STRESS_MIN       = 0.00
    NORMAL_STRESS_MAX       = 0.50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetConfig:
    BASE_MVA                = PowerSystemConfig.SN_MVA
    BASE_FREQUENCY          = PowerSystemConfig.BASE_FREQUENCY
    DEFAULT_NUM_NODES       = TopologyConfig.DEFAULT_NUM_NODES

    TRAIN_RATIO             = 0.70
    VAL_RATIO               = 0.15
    TEST_RATIO              = 0.15
    RATIO_TOLERANCE         = 1e-6      # Floating-point tolerance for ratio validation

    CASCADE_LABEL_THRESHOLD = 0.5       # Threshold to classify node as cascade
    AUGMENTATION_NOISE_STD  = 0.01      # Gaussian noise for training augmentation


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
    Simulation    = SimulationConfig
    Scenario      = ScenarioConfig
    Dataset       = DatasetConfig
