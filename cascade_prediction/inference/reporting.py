"""
Reporting
=========
Human-readable console report for cascade prediction results.
"""

from typing import Dict


def print_report(res: Dict, cascade_thresh: float, node_thresh: float):
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS (Scenario Analysis)")
    print("=" * 80)
    print(f"Inference Time: {res['inference_time']:.4f} seconds\n")

    gt = res["ground_truth"]
    pred = res["cascade_detected"]
    actual = gt["is_cascade"]

    print("--- 1. Overall Verdict ---")
    if pred and actual:         print("✅ Correctly detected a cascade.")
    elif not pred and not actual: print("✅ Correctly identified a normal scenario.")
    elif pred and not actual:   print("⚠️ FALSE POSITIVE (False Alarm)")
    else:                       print("❌ FALSE NEGATIVE (Missed Cascade)")

    print(f"Prediction: {pred} (Prob: {res['cascade_probability']:.3f} / Thresh: {cascade_thresh:.3f})")
    print(f"Ground Truth: {actual}")

    if actual or pred:
        print("\n--- 2. Node-Level Analysis ---")
        pred_nodes = set(res["high_risk_nodes"])
        actual_nodes = set(gt.get("failed_nodes", []))
        tp = len(pred_nodes & actual_nodes)
        fp = len(pred_nodes - actual_nodes)
        fn = len(actual_nodes - pred_nodes)
        print(f"Predicted Nodes at Risk: {len(pred_nodes)} (Thresh: {node_thresh:.3f})")
        print(f"Actual Failed Nodes:     {len(actual_nodes)}")
        print(f"  - Correctly Identified (TP): {tp}")
        print(f"  - Missed Nodes (FN):         {fn}")
        print(f"  - False Alarms (FP):         {fp}")

    if actual or pred:
        print("\n--- 3. Timing Analysis ---")
        pred_path = res["cascade_path"]
        act_path = gt.get("cascade_path", [])
        pred_times = [n["pred_time_minutes"] for n in pred_path] if pred_path else []
        act_times  = [x["time_minutes"] for x in act_path] if act_path else []
        min_pt, max_pt = (min(pred_times), max(pred_times)) if pred_times else (0.0, 0.0)
        min_at, max_at = (min(act_times),  max(act_times))  if act_times  else (0.0, 0.0)
        print(f"  {'Metric':<28} | {'Predicted':>12} | {'Actual':>12}")
        print(f"  {'-'*28}-+-{'-'*12}-+-{'-'*12}")
        print(f"  {'First failure (min)':<28} | {min_pt:>12.2f} | {min_at:>12.2f}")
        print(f"  {'Last failure (min)':<28} | {max_pt:>12.2f} | {max_at:>12.2f}")
        print(f"  {'Spread (min)':<28} | {max_pt-min_pt:>12.2f} | {max_at-min_at:>12.2f}")
        print(f"  {'Nodes at risk':<28} | {len(pred_path):>12} | {len(act_path):>12}")

    print("\n--- 4. Critical Information ---")
    print(f"System Frequency: {res['system_state']['frequency']:.2f} Hz")
    v_all = res["system_state"]["voltages"]
    if v_all:
        print(f"Voltage Range:    [{min(v_all):.3f}, {max(v_all):.3f}] p.u.")

    if pred and res["top_nodes"]:
        print("\nTop 5 High-Risk Nodes:")
        actual_nodes = set(gt.get("failed_nodes", []))
        for node in res["top_nodes"][:5]:
            nid = node["node_id"]
            status = "✓ (Actual)" if nid in actual_nodes else "✗ (Not Actual)"
            print(f"  - Node {nid:<3}: {node['score']:.4f} {status}")

    r = res["risk_assessment"]
    def get_lvl(s): return "(Critical)" if s > 0.8 else "(Severe)" if s > 0.6 else "(Medium)" if s > 0.3 else "(Low)"
    labels = ["Threat", "Vulnerability", "Impact", "Cascade Prob", "Response", "Safety", "Urgency"]
    print("\nAggregated Risk Assessment (7-Dimensions):")
    if len(r) >= 7:
        print("  - " + " | ".join(f"{l}: {s:.3f} {get_lvl(s):<10}" for l, s in zip(labels[:3], r[:3])))
        print("  - " + " | ".join(f"{l}: {s:.3f} {get_lvl(s):<10}" for l, s in zip(labels[3:6], r[3:6])))
        print(f"  - {labels[6]}: {r[6]:.3f} {get_lvl(r[6]):<10}")

    gt_risk = gt.get("ground_truth_risk", [])
    if gt_risk is not None and hasattr(gt_risk, "shape") and gt_risk.shape[1] >= 7:
        print("\n  Ground Truth Risk Assessment:")
        print("  - " + " | ".join(f"{l}: {s.mean():.3f} {get_lvl(s.mean()):<10}" for l, s in zip(labels[:3], gt_risk[:, :3])))
        print("  - " + " | ".join(f"{l}: {s.mean():.3f} {get_lvl(s.mean()):<10}" for l, s in zip(labels[3:6], gt_risk[:, 3:6])))
        print(f"  - {labels[6]}: {gt_risk[:,6].mean():.3f} {get_lvl(gt_risk[:,6].mean()):<10}")

    print("\n--- Risk Definitions ---")
    print("  Critical (0.8+): Immediate Failure | Severe (0.6+): High Danger | Medium (0.3+): Caution")
    print("  Dimensions: Threat (Stress), Vulnerability (Weakness), Impact (Consequence),")
    print("              Cascade Prob (Propagation), Urgency (Time Sensitivity).")

    print("\n--- 5. Cascade Path Analysis (Sequence Order) ---")
    pred_path = res["cascade_path"]
    actual_path = gt.get("cascade_path", [])
    pred_node_ids = set(n["node_id"] for n in pred_path)

    print(f"  {'Seq#':<5} | {'Pred Node':<10} | {'Prob':>6} | {'Pred(min)':>9} | {'Act Seq#':<8} | {'Act Node':<10} | {'Act(min)':>8} | {'Caught%':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    curr_act_seq = 0
    last_act_time = -999.0
    actual_seen = caught_so_far = 0

    for i in range(max(len(pred_path), len(actual_path))):
        p_seq = p_node = p_score = p_time = ""
        if i < len(pred_path):
            p = pred_path[i]
            p_seq, p_node, p_score, p_time = str(p["order"]), f"Node {p['node_id']}", f"{p['ranking_score']:.3f}", f"{p['pred_time_minutes']:.2f}"

        a_seq = a_node = a_time = caught_pct = ""
        if i < len(actual_path):
            a = actual_path[i]
            t = a["time_minutes"]
            if t > last_act_time + 0.1:
                curr_act_seq += 1
                last_act_time = t
            a_seq, a_node, a_time = str(curr_act_seq), f"Node {a['node_id']}", f"{t:.2f}"
            actual_seen += 1
            if a["node_id"] in pred_node_ids:
                caught_so_far += 1
            caught_pct = f"{caught_so_far / actual_seen * 100:.0f}%"

        print(f"  {p_seq:<5} | {p_node:<10} | {p_score:>6} | {p_time:>9} | {a_seq:<8} | {a_node:<10} | {a_time:>8} | {caught_pct:>8}")

    print("=" * 80 + "\n")
