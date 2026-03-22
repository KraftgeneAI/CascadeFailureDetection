"""
Deprecated Power Flow Functions Archive
========================================

This file contains deprecated power flow functions that were replaced by PyPSA-based
AC power flow in the main multimodal_data_generator.py file.

These functions are preserved for reference and potential rollback if needed.

Date Archived: 2026-02-16
Reason: Replaced by PyPSA AC power flow for accurate physics-based simulation
"""

import numpy as np
from typing import List, Optional, Tuple


# ====================================================================
# DEPRECATED FUNCTION 1: Simplified Non-Linear Power Flow
# ====================================================================
def _compute_simplified_power_flow(
    self,
    generation: np.ndarray, 
    load: np.ndarray,
    failed_lines: Optional[List[int]] = None,
    failed_nodes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    DEPRECATED: Use _compute_pypsa_power_flow instead.
    
    Compute simplified non-linear power flow for the grid.
    
    This function is deprecated in favor of PyPSA-based power flow
    which provides accurate AC power flow calculations based on proper
    physics equations.
    
    This is a SIMPLIFIED version of AC power flow that's fast and stable,
    but still captures the essential non-linear behavior that makes cascade
    prediction challenging.
    
    WHY SIMPLIFIED?
    ---------------
    Real AC power flow requires solving non-linear equations iteratively
    (Newton-Raphson method), which is slow and can fail to converge. For
    generating thousands of training scenarios, we need something faster
    and more stable.
    
    PHYSICS MODELS:
    ---------------
    1. VOLTAGE (Quadratic drop under load):
       V = 1.05 - 0.01×load - 0.04×load²
       - Light load (0.5): V ≈ 1.04 p.u. (normal)
       - Heavy load (1.0): V ≈ 1.00 p.u. (stressed)
       - Overload (1.2): V ≈ 0.94 p.u. (critical)
       Models accelerating voltage collapse under stress
    
    2. ANGLE (Cubic response to power):
       θ = 0.05×P + 0.01×P³
       - Generators (P > 0): Positive angle (leading)
       - Loads (P < 0): Negative angle (lagging)
       - Non-linear response captures instability
    
    3. LINE FLOW (Sine-based, like real AC):
       Flow = Susceptance × sin(θ_source - θ_dest) × 100
       - Approximates real AC power flow equation
       - Flow proportional to angle difference
       - Susceptance is line property (higher = more flow capacity)
    
    Parameters:
    -----------
    generation : np.ndarray, shape (num_nodes,)
        Power generation at each node in MW
        
    load : np.ndarray, shape (num_nodes,)
        Power consumption at each node in MW
        
    failed_lines : List[int], optional
        Indices of transmission lines that have failed (flow = 0)
        
    failed_nodes : List[int], optional
        Indices of nodes that have failed (generation = 0, load = 0)
    
    Returns:
    --------
    voltages : np.ndarray, shape (num_nodes,)
        Voltage magnitude at each node in per-unit (p.u.)
        Normal range: 0.95-1.05, Critical: <0.90
        
    angles : np.ndarray, shape (num_nodes,)
        Voltage angle at each node in radians
        Reference: Node 0 (slack bus) = 0.0
        
    line_flows : np.ndarray, shape (num_edges,)
        Active power flow on each line in MW
        Positive = source→destination, Negative = destination→source
        
    is_stable : bool
        Always True (this simplified model doesn't check stability)
        Real power flow can fail to converge (unstable system)
    """
    
    gen = generation.copy()
    ld = load.copy()
    if failed_nodes:
        for node in failed_nodes:
            gen[node] = 0.0
            ld[node] = 0.0
    
    # Net power injection at each bus
    P_net = gen - ld
    
    # 1. Model Voltages (Quadratic: y = 1.05 - ax - bx^2)
    # Models accelerating voltage drop under high load (collapse)
    load_norm = ld / (self.base_load + 1e-6) # Normalized load
    a = 0.01 # Linear drop
    b = 0.04 # Quadratic drop
    voltages = 1.05 - (a * load_norm) - (b * (load_norm**2)) + np.random.normal(0, 0.005, self.num_nodes)
    voltages = np.clip(voltages, 0.85, 1.05) # Clip to a reasonable range
    
    # 2. Model Angles (Cubic: y = ax + bx^3)
    # A simple non-linear response to power injection
    P_net_norm = P_net / (self.gen_capacity.max() + 1e-6) # Normalized power
    angles = (0.05 * P_net_norm) + (0.01 * (P_net_norm**3))
    angles[0] = 0.0 # Force slack bus (node 0) to be reference
    
    # 3. Model Line Flows (Sine-based: y = a*sin(x1-x2))
    # This is a simple analog to the real AC power flow equation
    src, dst = self.edge_index
    # Use angles as a proxy for the angular difference
    angle_diff = angles[src] - angles[dst]
    # Use line susceptance as the 'a' coefficient
    line_flows = self.line_susceptance * np.sin(angle_diff) * 100.0 # Scale factor
    line_flows += np.random.normal(0, 0.1, self.num_edges)
    
    # Handle failed lines
    if failed_lines:
        line_flows[failed_lines] = 0.0
    
    # 4. Model Stability
    # Always return True to prevent scenario rejection.
    is_stable = True
    
    return voltages, angles, line_flows, is_stable


# ====================================================================
# DEPRECATED FUNCTION 2: Realistic DC Power Flow
# ====================================================================
def _compute_realistic_power_flow(
    self, 
    generation: np.ndarray, 
    load: np.ndarray,
    failed_lines: Optional[List[int]] = None,
    failed_nodes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    DEPRECATED: Compute REALISTIC DC power flow with proper physics.
    Replaced by _compute_pypsa_power_flow for accurate AC power flow.
    Returns: voltages, angles, line_flows, is_stable
    """
    
    gen = generation.copy()
    ld = load.copy()
    if failed_nodes:
        for node in failed_nodes:
            gen[node] = 0.0
            ld[node] = 0.0
    
    # Net power injection at each bus
    P_net = gen - ld
    
    B = np.zeros((self.num_nodes, self.num_nodes))
    src, dst = self.edge_index
    
    active_lines = []
    for i in range(self.num_edges):
        if failed_lines is not None and i in failed_lines:
            continue
        active_lines.append(i)
        
        s, d = src[i].item(), dst[i].item()
        b = self.line_susceptance[i]
        
        # Build B matrix: B_ii = sum of susceptances, B_ij = -susceptance
        B[s, s] += b
        B[d, d] += b
        B[s, d] -= b
        B[d, s] -= b
    
    # Use slack bus (bus 0) as reference (theta_0 = 0)
    B_reduced = B[1:, 1:]
    P_reduced = P_net[1:]
    
    try:
        # Solve for voltage angles
        theta_reduced = np.linalg.solve(B_reduced, P_reduced)
        theta = np.zeros(self.num_nodes)
        theta[1:] = theta_reduced
        
        if np.max(np.abs(theta)) > np.radians(15):  # >15 degrees is unstable
            is_stable = False
        else:
            is_stable = True
            
    except np.linalg.LinAlgError:
        # Singular matrix = islanded system = unstable
        theta = np.zeros(self.num_nodes)
        is_stable = False
    
    line_flows = np.zeros(self.num_edges)
    for i in active_lines:
        s, d = src[i].item(), dst[i].item()
        line_flows[i] = self.line_susceptance[i] * (theta[s] - theta[d]) * 3.0
    
    # Voltage drops with heavy loading
    voltages = np.ones(self.num_nodes)
    for i in range(self.num_nodes):
        # Loading based on load and connections, plus some noise
        # Use degree from undirected graph for this heuristic
        num_connections = np.sum(self.adjacency_matrix[i]) 
        node_loading_factor = load[i] / (self.base_load[i] + 1e-6) * (1.0 + num_connections * 0.05)
        
        # Stability fix: less aggressive voltage drop
        voltage_drop = 0.05 * np.maximum(0, node_loading_factor - 0.7) 
            
        voltages[i] = 1.0 - voltage_drop + np.random.normal(0, 0.005)
    
    # Clip to realistic range
    voltages = np.clip(voltages, 0.85, 1.15)
    
    if np.any(voltages < 0.94) or np.any(voltages > 1.06):
        is_stable = False
    
    return voltages, theta, line_flows, is_stable


# ====================================================================
# DEPRECATED FUNCTION 3: Rule-Based Cascade Simulation
# ====================================================================
def _simulate_rule_based_cascade(
    self,
    stress_level: float,
    sequence_length: int = 60,
    target_failure_percentage: Optional[float] = None
) -> Tuple[List[int], List[float], List[str], int, str]:
    """
    DEPRECATED: This function uses a "forced" trigger.
    The new _generate_scenario_data is fully deterministic.
    
    This function was used to simulate cascade failures with a forced trigger
    mechanism, but has been replaced by a fully deterministic approach that
    checks actual failure thresholds.
    """
    pass


# ====================================================================
# NOTES ON REPLACEMENT
# ====================================================================
"""
These functions were replaced on 2026-02-16 with PyPSA-based AC power flow.

REPLACEMENT FUNCTION:
- _compute_pypsa_power_flow() in multimodal_data_generator.py

ADVANTAGES OF PYPSA:
1. Accurate AC power flow using Newton-Raphson solver
2. Proper handling of reactive power (Q)
3. Realistic voltage magnitudes and angles
4. Industry-standard convergence checking
5. Well-tested library used in power system research

ROLLBACK INSTRUCTIONS:
If you need to rollback to the simplified power flow:
1. Copy the desired function from this file back to multimodal_data_generator.py
2. Update the function calls from _compute_pypsa_power_flow to _compute_simplified_power_flow
3. Comment out the _initialize_pypsa_network() call in __init__
4. Update the initialization message

PERFORMANCE COMPARISON:
- Simplified: ~0.001s per power flow calculation
- PyPSA: ~0.01-0.05s per power flow calculation (10-50x slower but more accurate)

For large-scale data generation (10,000+ scenarios), consider the trade-off between
speed and accuracy based on your specific requirements.
"""
