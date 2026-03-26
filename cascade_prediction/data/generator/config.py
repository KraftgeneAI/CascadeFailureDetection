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
    FBETA                   = 0.5       # Beta for F-beta score (precision-focused)


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
    EDGE_FEATURES       = 7             # Edge feature vector length
    RISK_DIM            = 7             # Output dimension of risk assessment head
    GAT_DROPOUT         = 0.1           # Graph attention dropout
    LEAKY_RELU_SLOPE    = 0.2           # LeakyReLU negative slope in GAT


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
class LossConfig:
    # Default lambda weights (PhysicsInformedLoss constructor defaults)
    LAMBDA_PREDICTION   = 10.0
    LAMBDA_POWERFLOW    = 0.1
    LAMBDA_RISK         = 0.1
    # IMPROVED: timing lambda raised from 0.1 → 5.0 → 8.0.
    # v3 further increases it because:
    #  - Absolute-normalised targets (failure_time / DEFAULT_SEQ_LEN) give raw
    #    MSE of ~0.01-0.05, which is smaller than the prediction focal loss.
    #  - The new bias-correction and spread-enforcement terms are also small.
    #  - A weight of 8.0 brings the timing contribution to ~10-15% of total
    #    loss — sufficient to drive precise timing without hurting detection.
    LAMBDA_TIMING       = 8.0
    LAMBDA_ACTIVE_FLOW  = 0.1
    LAMBDA_TEMPERATURE  = 0.05
    LAMBDA_FREQUENCY    = 0.08
    LAMBDA_REACTIVE     = 0.1
    LAMBDA_VOLTAGE      = 1.0
    LAMBDA_CAPACITY     = 0.05

    # Focal loss parameters
    FOCAL_ALPHA         = 0.85          # PhysicsInformedLoss default
    FOCAL_GAMMA         = 2.0
    FOCAL_ALPHA_TRAIN   = 0.25          # train_model.py (calibrated path)
    FOCAL_ALPHA_FALLBACK = 0.15         # train_model.py (uncalibrated fallback)

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
    INFRA_SCADA_FEATURES    = 18        # SCADA measurement feature width
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

    # -- Node-level 119-feature MLP ------------------------------------------
    # Feature layout per node per timestep (119 total, IMPROVED v2):
    #   [0:18]   SCADA measurements          (18)
    #   [18:26]  PMU measurements             ( 8)
    #   [26:36]  Equipment status            (10)
    #   [36:37]  Active power injection       ( 1)
    #   [37:38]  Reactive power injection     ( 1)
    #   [38:76]  1-step temporal deltas of [0:38]  (38)
    #   [76:114] 2-step temporal deltas of [0:38]  (38)
    #   [114]    Normalised timestep position ( 1)
    #   [115]    TTF voltage  — normalised steps to voltage threshold  ( 1)
    #   [116]    TTF temp     — normalised steps to temperature threshold ( 1)
    #   [117]    TTF freq     — normalised steps to frequency threshold  ( 1)
    #   [118]    TTF loading  — normalised steps to loading threshold    ( 1)
    #
    # The 4 new TTF (time-to-failure) features encode physics-based estimates
    # of how many steps remain before each failure condition is breached.
    # They directly provide the timing signal needed by the TimingHead.
    NODE_FEATURE_BASE_DIM   = 38        # SCADA+PMU+equip+inj before deltas
    NODE_FEATURE_DIM        = 119       # Full per-node feature vector width (was 115)
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
    Simulation    = SimulationConfig
    Scenario      = ScenarioConfig
    Dataset       = DatasetConfig
    Training      = TrainingConfig
    Model         = ModelConfig
    Loss          = LossConfig
    Embedding     = EmbeddingConfig
