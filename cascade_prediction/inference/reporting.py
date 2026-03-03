"""
Reporting Module
===============
Provides functions for formatting and printing prediction reports.
"""

from typing import Dict, List


def format_risk_assessment(risk_scores: List[float]) -> str:
    """
    Format risk assessment scores into human-readable text.
    
    Args:
        risk_scores: List of 7 risk dimension scores
        
    Returns:
        Formatted risk assessment string
    """
    def get_level(score: float) -> str:
        if score > 0.8:
            return "(Critical)"
        elif score > 0.6:
            return "(Severe)"
        elif score > 0.3:
            return "(Medium)"
        else:
            return "(Low)"
    
    labels = [
        "Threat", "Vulnerability", "Impact", 
        "Cascade Prob", "Response", "Safety", "Urgency"
    ]
    
    if len(risk_scores) < 7:
        return "Insufficient risk data"
    
    lines = []
    lines.append("Aggregated Risk Assessment (7-Dimensions):")
    
    # First line: Threat, Vulnerability, Impact
    line1_parts = [
        f"{labels[i]}: {risk_scores[i]:.3f} {get_level(risk_scores[i]):<10}"
        for i in range(3)
    ]
    lines.append("  - " + " | ".join(line1_parts))
    
    # Second line: Cascade Prob, Response, Safety
    line2_parts = [
        f"{labels[i]}: {risk_scores[i]:.3f} {get_level(risk_scores[i]):<10}"
        for i in range(3, 6)
    ]
    lines.append("  - " + " | ".join(line2_parts))
    
    # Third line: Urgency
    lines.append(f"  - {labels[6]}: {risk_scores[6]:.3f} {get_level(risk_scores[6]):<10}")
    
    return "\n".join(lines)


def print_prediction_report(
    results: Dict,
    cascade_threshold: float,
    node_threshold: float
):
    """
    Print a comprehensive prediction report.
    
    Args:
        results: Prediction results dictionary
        cascade_threshold: Threshold for cascade detection
        node_threshold: Threshold for node failure detection
    """
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS (Scenario Analysis)")
    print("=" * 80)
    print(f"Inference Time: {results['inference_time']:.4f} seconds\n")
    
    # Overall verdict
    _print_overall_verdict(results, cascade_threshold)
    
    # Node-level analysis
    if results['ground_truth']['is_cascade'] or results['cascade_detected']:
        _print_node_analysis(results, node_threshold)
        _print_timing_analysis(results)
    
    # Critical information
    _print_critical_info(results)
    
    # Risk assessment
    _print_risk_assessment(results)
    
    # Cascade path analysis
    _print_cascade_path(results)
    
    # Risk definitions
    _print_risk_definitions()
    
    print("=" * 80 + "\n")


def _print_overall_verdict(results: Dict, cascade_threshold: float):
    """Print overall prediction verdict."""
    gt = results['ground_truth']
    pred = results['cascade_detected']
    actual = gt['is_cascade']
    
    print("--- 1. Overall Verdict ---")
    
    if pred and actual:
        print("✅ Correctly detected a cascade.")
    elif not pred and not actual:
        print("✅ Correctly identified a normal scenario.")
    elif pred and not actual:
        print("⚠️ FALSE POSITIVE (False Alarm)")
    elif not pred and actual:
        print("❌ FALSE NEGATIVE (Missed Cascade)")
    
    print(f"Prediction: {pred} (Prob: {results['cascade_probability']:.3f} / "
          f"Thresh: {cascade_threshold:.3f})")
    print(f"Ground Truth: {actual}\n")


def _print_node_analysis(results: Dict, node_threshold: float):
    """Print node-level analysis."""
    print("--- 2. Node-Level Analysis ---")
    
    pred_nodes = set(results['high_risk_nodes'])
    actual_nodes = set(results['ground_truth'].get('failed_nodes', []))
    
    tp = len(pred_nodes.intersection(actual_nodes))
    fp = len(pred_nodes - actual_nodes)
    fn = len(actual_nodes - pred_nodes)
    
    print(f"Predicted Nodes at Risk: {len(pred_nodes)} (Thresh: {node_threshold:.3f})")
    print(f"Actual Failed Nodes:     {len(actual_nodes)}")
    print(f"  - Correctly Identified (TP): {tp}")
    print(f"  - Missed Nodes (FN):         {fn}")
    print(f"  - False Alarms (FP):         {fp}\n")


def _print_timing_analysis(results: Dict):
    """Print timing analysis."""
    print("--- 3. Timing Analysis ---")
    print(f"  {'Metric':<28} | {'Predicted':<17} | {'Ground Truth':<17}")
    print(f"  {'-'*28}|{'-'*18}|{'-'*18}")
    
    # Predicted timing
    scores = [n['score'] for n in results['top_nodes']]
    min_s, max_s = (min(scores), max(scores)) if scores else (0.0, 0.0)
    score_spread = max_s - min_s
    
    # Actual timing
    act_path = results['ground_truth'].get('cascade_path', [])
    min_t, max_t = 0.0, 0.0
    if act_path:
        times = [x['time_minutes'] for x in act_path]
        min_t, max_t = min(times), max(times)
    
    print(f"  {'Prediction Mode':<28} | {'Relative Rank':<17} | {'Absolute Time':<17}")
    print(f"  {'Range (Start -> End)':<28} | {max_s:.3f} -> {min_s:.3f}   | "
          f"{min_t:.2f} -> {max_t:.2f} min")
    print(f"  {'Sequence Spread':<28} | {score_spread:.3f} (Score)   | "
          f"{max_t - min_t:.2f} minutes\n")


def _print_critical_info(results: Dict):
    """Print critical system information."""
    print("--- 4. Critical Information ---")
    print(f"System Frequency: {results['system_state']['frequency']:.2f} Hz")
    
    v_all = results['system_state']['voltages']
    if v_all:
        print(f"Voltage Range:    [{min(v_all):.3f}, {max(v_all):.3f}] p.u.")
    
    if results['cascade_detected'] and results['top_nodes']:
        print("\nTop 5 High-Risk Nodes:")
        actual_nodes = set(results['ground_truth'].get('failed_nodes', []))
        for node in results['top_nodes'][:5]:
            nid = node['node_id']
            status = "✓ (Actual)" if nid in actual_nodes else "✗ (Not Actual)"
            print(f"  - Node {nid:<3}: {node['score']:.4f} {status}")
    
    print()


def _print_risk_assessment(results: Dict):
    """Print risk assessment."""
    risk_text = format_risk_assessment(results['risk_assessment'])
    print(risk_text)
    
    # Ground truth risk
    gt_risk = results['ground_truth'].get('ground_truth_risk', [])
    if gt_risk and len(gt_risk) >= 7:
        print("\n  Ground Truth Risk Assessment:")
        gt_text = format_risk_assessment(gt_risk)
        # Print without the header
        for line in gt_text.split('\n')[1:]:
            print(line)
    
    print()


def _print_cascade_path(results: Dict):
    """Print cascade path analysis."""
    print("--- 5. Cascade Path Analysis (Sequence Order) ---")
    
    pred_path = results['cascade_path']
    actual_path = results['ground_truth'].get('cascade_path', [])
    
    print(f"  {'Seq #':<6} | {'Predicted Node':<15} | {'Score':<8} | "
          f"{'Actual Seq #':<15} | {'Actual Node':<15} | {'Delta T (min)':<15}")
    print(f"  {'-'*6} | {'-'*15} | {'-'*8} | {'-'*15} | {'-'*15} | {'-'*15}")
    
    max_rows = max(len(pred_path), len(actual_path))
    curr_act_seq = 0
    last_act_time = -999.0
    
    for i in range(max_rows):
        # Predicted path
        p_seq, p_node, p_score = "", "", ""
        if i < len(pred_path):
            p_item = pred_path[i]
            p_seq = str(p_item['order'])
            p_node = f"Node {p_item['node_id']}"
            p_score = f"{p_item['ranking_score']:.3f}"
        
        # Actual path
        a_seq, a_node, a_time = "", "", ""
        if i < len(actual_path):
            a_item = actual_path[i]
            t = a_item['time_minutes']
            if t > last_act_time + 0.1:
                curr_act_seq += 1
                last_act_time = t
            a_seq = str(curr_act_seq)
            a_node = f"Node {a_item['node_id']}"
            a_time = f"{t:.2f}"
        
        print(f"  {p_seq:<6} | {p_node:<15} | {p_score:<8} | "
              f"{a_seq:<15} | {a_node:<15} | {a_time:<15}")


def _print_risk_definitions():
    """Print risk level definitions."""
    print("\n--- Risk Definitions ---")
    print("  Critical (0.8+): Immediate Failure | Severe (0.6+): High Danger | "
          "Medium (0.3+): Caution")
    print("  Dimensions: Threat (Stress), Vulnerability (Weakness), Impact (Consequence),")
    print("              Cascade Prob (Propagation), Urgency (Time Sensitivity).")
