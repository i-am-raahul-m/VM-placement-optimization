import pandas as pd
import numpy as np

def prioritize_machines(
    safe_machines_df,
    reqs,
    alpha=0.5,
    ram_power_per_gb=None,
    gpu_vram_power_per_gb=None,
    cpu_power_per_core=None,
    gpu_power_per_core=None,
    eps=1e-9
):
    """
    Rank safe machines by combined score of (energy impact, best-fit).
    
    Args:
      safe_machines_df: pd.DataFrame with one row per machine. MUST contain columns:
         - cpu_cores_total, cpu_watts, gpu_cores_total, gpu_vram_total_gb, gpu_watts,
           ram_total_gb, cpu_cores_used, gpu_cores_used, gpu_vram_used_gb, ram_used_gb,
           current_cpu_power_w, current_gpu_power_w
      reqs: dict with requested resources:
         {'req_cpu_cores', 'req_gpu_cores', 'req_gpu_vram_gb', 'req_ram_gb'}
      alpha: float in [0,1] - weight for energy vs fit. final_score = alpha*energy + (1-alpha)*fit
      *_power_per_* : optional overrides for per-unit power estimates. If None, derive from machine totals.
      eps: small number to avoid divide-by-zero.
      
    Returns:
      ranked_df: copy of safe_machines_df with added columns:
         remaining_cpu_cores, remaining_gpu_cores, remaining_gpu_vram_gb, remaining_ram_gb,
         fit_score (0..inf, lower better), est_incremental_power_w, energy_score (0..1),
         final_score (0..1, lower better). Sorted by final_score ascending.
      best_machine_row: Series of top machine (or None if none feasible).
    """

    # Get required columns
    df = safe_machines_df.copy().reset_index(drop=False)  
    required_cols = [
        "cpu_cores_total", "cpu_watts", "gpu_cores_total", "gpu_vram_total_gb", "gpu_watts",
        "ram_total_gb", "cpu_cores_used", "gpu_cores_used", "gpu_vram_used_gb", "ram_used_gb",
        "current_cpu_power_w", "current_gpu_power_w"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in safe_machines_df: {missing}")

    # requested resources
    r_cpu = float(reqs.get("req_cpu_cores", 0.0))
    r_gpu = float(reqs.get("req_gpu_cores", 0.0))
    r_vram = float(reqs.get("req_gpu_vram_gb", 0.0))
    r_ram = float(reqs.get("req_ram_gb", 0.0))

    # Remaining capacities
    df["remaining_cpu_cores"] = df["cpu_cores_total"] - df["cpu_cores_used"]
    df["remaining_gpu_cores"] = df["gpu_cores_total"] - df["gpu_cores_used"]
    df["remaining_gpu_vram_gb"] = df["gpu_vram_total_gb"] - df["gpu_vram_used_gb"]
    df["remaining_ram_gb"] = df["ram_total_gb"] - df["ram_used_gb"]

    # Filter infeasible
    feasible_mask = (
        (df["remaining_cpu_cores"] + eps >= r_cpu) &
        (df["remaining_gpu_cores"] + eps >= r_gpu) &
        (df["remaining_gpu_vram_gb"] + eps >= r_vram) &
        (df["remaining_ram_gb"] + eps >= r_ram)
    )
    df = df[feasible_mask].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(), None


    # -------------------------
    # Fit score (how tightly the request fits into remaining resources)
    # smaller is better. We use normalized absolute leftover fraction per resource
    # -------------------------

    # Avoid dividing by zero by using totals where appropriate, fallback to 1
    df["fit_cpu"]  = np.abs(df["remaining_cpu_cores"] - r_cpu) / (df["cpu_cores_total"].replace(0, eps))
    df["fit_gpu"]  = np.abs(df["remaining_gpu_cores"] - r_gpu) / (df["gpu_cores_total"].replace(0, eps))
    # For vram & ram use totals as denom
    df["fit_vram"] = np.abs(df["remaining_gpu_vram_gb"] - r_vram) / (df["gpu_vram_total_gb"].replace(0, eps))
    df["fit_ram"]  = np.abs(df["remaining_ram_gb"] - r_ram) / (df["ram_total_gb"].replace(0, eps))

    df["fit_score_raw"] = (df[["fit_cpu","fit_gpu","fit_vram","fit_ram"]].sum(axis=1)) / 4.0

    min_fit = df["fit_score_raw"].min()
    max_fit = df["fit_score_raw"].max()
    if abs(max_fit - min_fit) < eps:
        df["fit_score"] = 0.0
    else:
        df["fit_score"] = (df["fit_score_raw"] - min_fit) / (max_fit - min_fit)

    # Energy estimation
    # We estimate incremental power (watts) caused by allocating the requested resources.
    # By default we derive per-unit powers from machine totals (cpu_watts / cpu_cores_total, etc.)
    # But user can override with cpu_power_per_core, gpu_power_per_core, gpu_vram_power_per_gb, ram_power_per_gb
    
    def derive_row_power(row):
        cpu_per_core = cpu_power_per_core if cpu_power_per_core is not None else \
            (row["cpu_watts"] / max(row["cpu_cores_total"], eps))
        gpu_per_core = gpu_power_per_core if gpu_power_per_core is not None else \
            (row["gpu_watts"] / max(row["gpu_cores_total"], eps))
        gpu_vram_per_gb = gpu_vram_power_per_gb if gpu_vram_power_per_gb is not None else \
            (row["gpu_watts"] / max(row["gpu_vram_total_gb"], eps))
        if ram_power_per_gb is not None:
            ram_per_gb = ram_power_per_gb
        else:
            
            cur_power = float(row.get("current_cpu_power_w", 0.0)) + float(row.get("current_gpu_power_w", 0.0))
            total_ram_gb = max(row["ram_total_gb"], 1.0)
           
            ram_per_gb = max(0.05, min(2.0, (cur_power * 0.02) / total_ram_gb))  # between 0.05 and 2 W/GB heuristic
        
        inc_power = cpu_per_core * r_cpu + gpu_per_core * r_gpu + gpu_vram_per_gb * r_vram + ram_per_gb * r_ram
        return inc_power

    df["est_incremental_power_w"] = df.apply(derive_row_power, axis=1)

    min_e = df["est_incremental_power_w"].min()
    max_e = df["est_incremental_power_w"].max()
    if abs(max_e - min_e) < eps:
        df["energy_score"] = 0.0
    else:
        df["energy_score"] = (df["est_incremental_power_w"] - min_e) / (max_e - min_e)

    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")

    df["final_score"] = alpha * df["energy_score"] + (1.0 - alpha) * df["fit_score"]

    ranked = df.sort_values(["final_score", "est_incremental_power_w", "fit_score"]).reset_index(drop=True)

    best_row = ranked.iloc[0] if ranked.shape[0] > 0 else None
    return ranked, best_row


if __name__ == '__main__':
    machines_df = pd.read_csv("data/model_features_test.csv")

    requested_vm = {
        "req_cpu_cores": 4,
        "req_gpu_cores": 0,
        "req_gpu_vram_gb": 0.0,
        "req_ram_gb": 8.0
    }

    ranked, best = prioritize_machines(
        machines_df,
        requested_vm,
        alpha=0.4,
        ram_power_per_gb=0.2
    )

    if ranked.empty:
        print("No feasible machines found.")
    else:
        print("Top candidates (best first):")
        display_columns = [
            "index",
            "remaining_cpu_cores","remaining_gpu_cores","remaining_gpu_vram_gb","remaining_ram_gb",
            "est_incremental_power_w","fit_score","energy_score","final_score"
        ]
        print(ranked[display_columns].head(10))
        print("\nBest machine (chosen):")
        print(best[display_columns])
