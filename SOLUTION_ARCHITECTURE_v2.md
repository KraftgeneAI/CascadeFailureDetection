# CASCADE FAILURE DETECTION SYSTEM - SOLUTION ARCHITECTURE
## End-to-End System Design

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CASCADE FAILURE PREDICTION SYSTEM                          ║
║                    Multi-Modal AI for Power Grid Safety                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA ACQUISITION LAYER                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PHYSICAL GRID                                                               │
│  ─────────────                                                               │
│  IEEE 118-Bus Power System                                                   │
│  • 118 nodes (generators, substations, loads)                               │
│  • 186 transmission lines (edges)                                            │
│  • 230 kV nominal voltage                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
        ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
        │ ENVIRONMENTAL │  │ INFRASTRUCTURE│  │   ROBOTIC    │
        │   SENSORS     │  │    SENSORS    │  │   SENSORS    │
        └───────────────┘  └──────────────┘  └──────────────┘
               │                  │                   │
               ▼                  ▼                   ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │ • Satellite      │ │ • SCADA          │ │ • Visual Camera  │
    │   [12×16×16]     │ │   [13 features]  │ │   [3×32×32]      │
    │ • Weather        │ │ • PMU            │ │ • Thermal Camera │
    │   [80 features]  │ │   [8 features]   │ │   [1×32×32]      │
    │ • Threat Detect  │ │ • Equipment      │ │ • Vibration/     │
    │   [6 features]   │ │   [10 features]  │ │   Acoustic [12]  │
    └──────────────────┘ └──────────────────┘ └──────────────────┘
               │                  │                   │
               └──────────────────┼───────────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │  Multi-Modal Data Stream     │
                    │  [B, T=30, N=118, Features]  │
                    └──────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                      2. DATA GENERATION & STORAGE LAYER                       │
└──────────────────────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────┐
        │           DATA GENERATION PIPELINE                 │
        │           ────────────────────────                 │
        │                                                    │
        │  ┌──────────────────┐      ┌──────────────────┐  │
        │  │ LEGACY HEURISTIC │      │  DIGITAL TWIN    │  │
        │  │    GENERATOR     │ OR   │  (PyPSA AC PF)   │  │
        │  │                  │      │                  │  │
        │  │ • Fast (1s/scen) │      │ • Slow (5s/scen) │  │
        │  │ • Approximations │      │ • Physics-exact  │  │
        │  │ • No Q data      │      │ • Full V,θ,P,Q   │  │
        │  └──────────────────┘      └──────────────────┘  │
        │           │                         │             │
        │           └────────┬────────────────┘             │
        │                    ▼                              │
        │         Scenario Generation                       │
        │         ──────────────────                        │
        │         • Normal (30%)                            │
        │         • Cascade (60%)                           │
        │         • Stressed (5%)                           │
        │         • Near-miss (5%)                          │
        └────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────┐
        │        DATA STORAGE (Train/Val/Test Split)      │
        │        ─────────────────────────────────────    │
        │                                                 │
        │  ./data/train/   (70%)  ~7,000 scenarios       │
        │  ./data/val/     (15%)  ~1,500 scenarios       │
        │  ./data/test/    (15%)  ~1,500 scenarios       │
        │                                                 │
        │  Format: PyTorch tensors + metadata JSON       │
        │  Size: ~50 MB per scenario                     │
        └─────────────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────┐
        │         CascadeDataset (PyTorch)                │
        │         ────────────────────────                │
        │  • Lazy loading (1 file at a time)              │
        │  • Dynamic truncation (variable T)              │
        │  • On-the-fly normalization                     │
        │  • Batch collation with padding                 │
        └─────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         3. MODEL ARCHITECTURE LAYER                           │
└──────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │       UNIFIED CASCADE PREDICTION MODEL (806K params)        │
        │       ───────────────────────────────────────────────       │
        └─────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼──────────────────────────┐
        ▼                           ▼                          ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Environmental   │      │ Infrastructure   │      │   Robotic       │
│  Embedding      │      │   Embedding      │      │  Embedding      │
│                 │      │                  │      │                 │
│ Sat → CNN       │      │ SCADA → MLP      │      │ Visual → CNN    │
│ Weather → LSTM  │      │ PMU → MLP        │      │ Thermal → CNN   │
│ Threat → MLP    │      │ Equip → MLP      │      │ Sensor → MLP    │
│                 │      │                  │      │                 │
│ Output: [B,T,   │      │ Output: [B,T,    │      │ Output: [B,T,   │
│         N,128]  │      │         N,128]   │      │         N,128]  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
        │                         │                          │
        └─────────────────────────┼──────────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │  Multi-Modal Fusion      │
                    │  ──────────────────      │
                    │  MultiheadAttention(4)   │
                    │  Cross-modal interaction │
                    │  Output: [B,T,N,128]     │
                    └──────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │         TEMPORAL-SPATIAL GNN (Hybrid)               │
        │         ────────────────────────────                │
        │                                                     │
        │  Per Timestep (t = 0..T-1):                        │
        │  ┌─────────────────────────────────────────┐       │
        │  │  1. Graph Attention (GAT)               │       │
        │  │     • Message passing on edges          │       │
        │  │     • Spatial aggregation               │       │
        │  │                                          │       │
        │  │  2. LSTM (2 layers)                     │       │
        │  │     • Short-range temporal memory       │       │
        │  │                                          │       │
        │  │  3. Transformer (2 layers)              │       │
        │  │     • Long-range cascade dependencies   │       │
        │  │     • Causal attention mask             │       │
        │  │                                          │       │
        │  │  4. Adaptive Blend Gate                 │       │
        │  │     • gate = σ(MLP([lstm, xformer]))   │       │
        │  │     • h = gate·lstm + (1-gate)·xformer │       │
        │  └─────────────────────────────────────────┘       │
        │                                                     │
        │  Output: h_states [B, N, T, 128]                   │
        │  Final: h = h_states[:,:,-1,:] [B, N, 128]        │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │      Additional GNN Layers (3 layers)               │
        │      Spatial refinement via GAT + residual          │
        │      Output: h [B, N, 128]                          │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │              PREDICTION HEADS                        │
        │              ────────────────                        │
        │                                                      │
        │  ┌────────────────────────────────────────┐         │
        │  │ Stage 1: Primary Predictions           │         │
        │  │ ───────────────────────────            │         │
        │  │ • failure_prob [B,N,1]    ← Binary     │         │
        │  │ • failure_timing [B,N,1]  ← Sequence   │         │
        │  │ • risk_scores [B,N,7]     ← Risk vec   │         │
        │  │ • voltages [B,N,1]        ← Physics    │         │
        │  │ • angles [B,N,1]          ← Physics    │         │
        │  │ • line_flows [B,E,1]      ← Physics    │         │
        │  │ • reactive_flows [B,E,1]  ← Physics    │         │
        │  │ • frequency [B,1,1]       ← Physics    │         │
        │  │ • temperature [B,N,1]     ← Physics    │         │
        │  └────────────────────────────────────────┘         │
        │                      │                               │
        │                      ▼                               │
        │  ┌────────────────────────────────────────┐         │
        │  │ Stage 2: Refinement Head               │         │
        │  │ ────────────────────────               │         │
        │  │ For borderline nodes (0.20-0.55):      │         │
        │  │   • Gather 1-hop + 2-hop context       │         │
        │  │   • MLP refinement                     │         │
        │  │   • Blend: 0.4·stage1 + 0.6·refined   │         │
        │  │                                         │         │
        │  │ Reduces false positives: 27% → <15%   │         │
        │  └────────────────────────────────────────┘         │
        └─────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          4. TRAINING & OPTIMIZATION LAYER                     │
└──────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │         PHYSICS-INFORMED LOSS FUNCTION              │
        │         ──────────────────────────────              │
        │                                                     │
        │  L_total = L_pred + Σ λ_i · L_physics_i            │
        │                                                     │
        │  ┌──────────────────────────────────────────┐      │
        │  │ 1. Prediction Loss (Focal, α=0.15)       │      │
        │  │    Binary cascade detection              │      │
        │  │    Weight: 1.0 (always dominant)         │      │
        │  └──────────────────────────────────────────┘      │
        │                                                     │
        │  ┌──────────────────────────────────────────┐      │
        │  │ 2. Timing Loss (Adaptive Ranking) ★      │      │
        │  │    • Pairwise margin ranking             │      │
        │  │    • Score separation penalty            │      │
        │  │    • Anchor loss (endpoints)             │      │
        │  │    Weight: λ_timing ≥ 0.30 (floor)      │      │
        │  └──────────────────────────────────────────┘      │
        │                                                     │
        │  ┌──────────────────────────────────────────┐      │
        │  │ 3. Risk Loss (MSE on 7D vector)          │      │
        │  │    Weight: λ_risk ≥ 0.15 (floor)         │      │
        │  └──────────────────────────────────────────┘      │
        │                                                     │
        │  ┌──────────────────────────────────────────┐      │
        │  │ 4. Physics Losses                        │      │
        │  │    • Powerflow MSE (λ × 0.1)            │      │
        │  │    • Voltage MSE (λ × 0.1)              │      │
        │  │    • Reactive MSE (λ × 0.1)             │      │
        │  │    • Frequency MSE (λ × 0.1)            │      │
        │  │    • Temperature MSE (λ × 0.1)          │      │
        │  │    • Angle MSE (λ × 0.1) ★ NEW          │      │
        │  └──────────────────────────────────────────┘      │
        │                                                     │
        │  ┌──────────────────────────────────────────┐      │
        │  │ 5. Temporal Consistency Loss ★ NEW       │      │
        │  │    Couples physics to timing:            │      │
        │  │    If t_A < t_B → V_A ≤ V_B             │      │
        │  │    Weight: 0.05                          │      │
        │  └──────────────────────────────────────────┘      │
        │                                                     │
        │  ★ = New improvements (6-point roadmap)            │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │      DYNAMIC LOSS WEIGHT CALIBRATION                │
        │      ────────────────────────────                   │
        │  1. Run 20 batches with all λ = 1.0                 │
        │  2. Measure raw loss magnitude per component        │
        │  3. Scale weights to equalize magnitudes            │
        │  4. Apply floors: λ_timing ≥ 0.30, λ_risk ≥ 0.15   │
        │  5. Scale physics × 0.1 (prediction-first)          │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │           OPTIMIZER & TRAINING                      │
        │           ─────────────────                         │
        │  • AdamW (lr=1e-4, weight_decay=1e-4)               │
        │  • Gradient clipping (max_norm=20)                  │
        │  • ReduceLROnPlateau (patience=5, factor=0.5)       │
        │  • Early stopping (patience=15)                     │
        │  • Batch size: 4 (CPU) or 32 (GPU)                  │
        │  • Epochs: 30-50                                    │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │          CHECKPOINTING & TRACKING                   │
        │          ────────────────────────                   │
        │  Saved:                                             │
        │  • best_model.pth (lowest val loss)                 │
        │  • best_timing_model.pth (lowest timing MAE)        │
        │  • best_f1_model.pth (highest F1)                   │
        │  • checkpoint_epoch_N.pth (periodic)                │
        │                                                     │
        │  History (per epoch):                               │
        │  • Train/Val loss, F1, accuracy, MAE, MSE           │
        │  • Learning rate schedule                           │
        └─────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        5. INFERENCE & EVALUATION LAYER                        │
└──────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │              INFERENCE PIPELINE                     │
        │              ──────────────────                     │
        │                                                     │
        │  Input: Test scenario (30 timesteps)                │
        │         ↓                                           │
        │  Load: best_model.pth checkpoint                    │
        │         ↓                                           │
        │  Forward: model(scenario) → predictions             │
        │         ↓                                           │
        │  Output:                                            │
        │    • Binary: Will cascade occur? (prob)             │
        │    • Sequence: Failure order [node_ids, times]      │
        │    • Risk: 7D vector per node                       │
        │    • Physics: V, θ, P, Q, f at each node/edge       │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │              EVALUATION METRICS                     │
        │              ──────────────────                     │
        │                                                     │
        │  Cascade Detection:                                 │
        │    • Accuracy, Precision, Recall, F1                │
        │    • ROC-AUC, Confusion Matrix                      │
        │                                                     │
        │  Node Detection:                                    │
        │    • Per-node F1, Accuracy                          │
        │    • False Positive Rate (target: <15%)             │
        │                                                     │
        │  Timing Accuracy:                                   │
        │    • Mean Absolute Error (target: <2.5s)            │
        │    • Spearman correlation (sequence order)          │
        │    • Score spread (target: >0.15)                   │
        │                                                     │
        │  Physics Accuracy:                                  │
        │    • Voltage MAE, Angle MAE, Frequency MAE          │
        │    • Powerflow RMSE, Reactive RMSE                  │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │            REPORTING & VISUALIZATION                │
        │            ────────────────────────                 │
        │                                                     │
        │  inference_report.txt:                              │
        │    • Per-scenario predictions vs ground truth       │
        │    • Confusion matrices                             │
        │    • Error distributions                            │
        │    • Physics compliance checks                      │
        │                                                     │
        │  training_curves.png:                               │
        │    • Loss vs epoch                                  │
        │    • F1 vs epoch                                    │
        │    • Timing MAE vs epoch                            │
        └─────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        6. DEPLOYMENT & MONITORING LAYER                       │
└──────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │            PRODUCTION DEPLOYMENT                    │
        │            ────────────────────                     │
        │                                                     │
        │  ┌────────────────────────────────────┐             │
        │  │ Option A: Batch Processing         │             │
        │  │ ────────────────────────           │             │
        │  │ • Cron job: Run every hour         │             │
        │  │ • Process: Latest grid data        │             │
        │  │ • Output: Risk dashboard update    │             │
        │  └────────────────────────────────────┘             │
        │                                                     │
        │  ┌────────────────────────────────────┐             │
        │  │ Option B: Real-Time API            │             │
        │  │ ────────────────────────           │             │
        │  │ • Flask/FastAPI endpoint           │             │
        │  │ • Input: Live sensor stream        │             │
        │  │ • Output: Cascade risk alert       │             │
        │  │ • Latency: <1 second               │             │
        │  └────────────────────────────────────┘             │
        │                                                     │
        │  ┌────────────────────────────────────┐             │
        │  │ Option C: Edge Deployment          │             │
        │  │ ────────────────────────           │             │
        │  │ • On-site inference (substation)   │             │
        │  │ • ONNX model export                │             │
        │  │ • Low-power hardware (CPU)         │             │
        │  └────────────────────────────────────┘             │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │          MONITORING & ALERTING                      │
        │          ─────────────────────                      │
        │                                                     │
        │  Dashboard (real-time):                             │
        │    • Current grid state (V, P, Q, f)                │
        │    • Cascade risk score (0-100%)                    │
        │    • Predicted failure sequence                     │
        │    • Alert status (green/yellow/red)                │
        │                                                     │
        │  Alerts:                                            │
        │    • Email/SMS if risk > 80%                        │
        │    • Escalation if cascade detected                 │
        │    • Operator console notification                  │
        │                                                     │
        │  Logging:                                           │
        │    • All predictions (audit trail)                  │
        │    • Model performance drift                        │
        │    • False alarm rate tracking                      │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │           MODEL MAINTENANCE                         │
        │           ─────────────────                         │
        │                                                     │
        │  Retraining:                                        │
        │    • Collect new failure events                     │
        │    • Retrain every 3-6 months                       │
        │    • A/B test new vs old model                      │
        │                                                     │
        │  Validation:                                        │
        │    • Compare predictions to actual outcomes         │
        │    • Track F1, false alarm rate over time           │
        │    • Flag performance degradation                   │
        │                                                     │
        │  Updates:                                           │
        │    • Physics improvements (digital twin)            │
        │    • New sensor modalities                          │
        │    • Architecture refinements                       │
        └─────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                            SYSTEM SPECIFICATIONS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE REQUIREMENTS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Accuracy:                                                                   │
│    • Cascade Detection F1 > 0.90                                            │
│    • Node Detection F1 > 0.85                                               │
│    • False Positive Rate < 15%                                              │
│    • Timing MAE < 2.5 seconds                                               │
│                                                                              │
│  Speed:                                                                      │
│    • Inference: <1 second per scenario (CPU)                                │
│    • Batch processing: >100 scenarios/minute                                │
│    • Training: ~2 hours/epoch (5K scenarios, CPU)                           │
│                                                                              │
│  Resource Usage:                                                             │
│    • Model size: 3.1 MB (806K parameters)                                   │
│    • Training RAM: ~2.4 GB (batch_size=4, CPU)                              │
│    • Inference RAM: ~1.8 GB                                                 │
│    • Storage: ~5 GB (10K scenarios)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA SPECIFICATIONS                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Format:                                                               │
│    • Temporal: 30 timesteps (can vary 0.3x to 0.85x via truncation)         │
│    • Spatial: 118 nodes, 186 edges                                          │
│    • Multi-modal: 9 sensor types (environmental, infra, robotic)            │
│    • Total features: ~15K per timestep (mostly images)                      │
│                                                                              │
│  Output Format:                                                              │
│    • Binary: Cascade yes/no (1 scalar)                                      │
│    • Timing: Per-node failure sequence (118 scalars)                        │
│    • Risk: 7D vector per node (118 × 7)                                     │
│    • Physics: V, θ, P, Q, f (118 nodes + 186 edges)                         │
│                                                                              │
│  Scenario Types:                                                             │
│    • Normal: 30% (stable operation)                                          │
│    • Cascade: 60% (2-20 nodes fail)                                         │
│    • Stressed: 5% (high load, stable)                                       │
│    • Near-miss: 5% (cascade arrests at 2-3 nodes)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TECHNOLOGY STACK                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Core:                                                                       │
│    • PyTorch 2.x (deep learning framework)                                  │
│    • Python 3.13                                                            │
│    • NumPy, SciPy (numerical computing)                                     │
│                                                                              │
│  Physics:                                                                    │
│    • PyPSA (AC power flow solver) ← Digital Twin                            │
│    • NetworkX (graph operations)                                            │
│    • Pandas (data wrangling)                                                │
│                                                                              │
│  Hardware:                                                                   │
│    • Training: CPU (8 GB RAM laptop) or GPU (optional)                      │
│    • Inference: CPU (edge deployment capable)                               │
│                                                                              │
│  Deployment:                                                                 │
│    • Flask/FastAPI (API server)                                             │
│    • ONNX (model export for edge)                                           │
│    • Docker (containerization)                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                            KEY INNOVATIONS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. HYBRID TEMPORAL ENCODER (LSTM + Transformer)
   ─────────────────────────────────────────────
   Problem: LSTM-only misses long-range cascade dependencies (A→B→C→D chains)
   Solution: Parallel Transformer path with causal attention + learned blend gate
   Impact: Captures multi-hop failure propagation, improves sequence accuracy

2. ADAPTIVE-MARGIN RANKING LOSS
   ─────────────────────────────
   Problem: Score compression (predictions span only 0.05 range)
   Solution: Dynamic margin based on time gap + separation penalty + anchors
   Impact: Score spread 0.05 → 0.15+, better sequence ordering

3. TWO-STAGE REFINEMENT HEAD
   ────────────────────────────
   Problem: High false positive rate (27%)
   Solution: Re-examine borderline predictions with 2-hop neighborhood context
   Impact: FP rate reduced to <15% without hurting recall

4. DIGITAL TWIN INTEGRATION (PyPSA)
   ──────────────────────────────────
   Problem: Heuristic data violates Kirchhoff's laws, "Ghost Heads" disabled
   Solution: Replace generator with AC power flow solver for V, θ, P, Q
   Impact: Physics-compliant training, reactive power head activated

5. TEMPORAL CONSISTENCY PHYSICS LOSS
   ────────────────────────────────────
   Problem: Physics predictions inconsistent with timing predictions
   Solution: Couple voltage degradation to failure sequence ordering
   Impact: Physically plausible predictions, prevents hallucinations

6. DYNAMIC LOSS WEIGHT CALIBRATION
   ────────────────────────────────
   Problem: Some losses dominate, others starve (timing/risk underweighted)
   Solution: Auto-calibrate weights + hard floors to prevent starvation
   Impact: All heads train effectively, no gradient imbalance

╔══════════════════════════════════════════════════════════════════════════════╗
║                          DEPLOYMENT SCENARIOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SCENARIO A: Grid Operations Center (Real-Time Monitoring)
──────────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  Live SCADA Feed (every 1 min)                         │
│         ↓                                               │
│  Model Inference (<1 sec)                              │
│         ↓                                               │
│  Dashboard Update (risk score, predicted sequence)     │
│         ↓                                               │
│  Alert if risk > 80% (email/SMS to operators)          │
└────────────────────────────────────────────────────────┘

SCENARIO B: Contingency Planning (Offline Analysis)
────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  What-if scenarios (e.g., "What if line 42 fails?")   │
│         ↓                                               │
│  Generate synthetic data (digital twin)                │
│         ↓                                               │
│  Batch inference (100 scenarios)                       │
│         ↓                                               │
│  Report: Which scenarios cascade? Severity ranking     │
└────────────────────────────────────────────────────────┘

SCENARIO C: Substation Edge Deployment (Autonomous Protection)
───────────────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  Local sensors (PMU, SCADA) at substation              │
│         ↓                                               │
│  On-device inference (ONNX model, <500ms)              │
│         ↓                                               │
│  If cascade detected → Auto-trip circuit breakers      │
│         ↓                                               │
│  Log event + notify control center                     │
└────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                          END OF ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════
```
