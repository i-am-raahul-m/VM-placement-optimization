import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
from pathlib import Path

# ------------------------
# Load multiple models
# ------------------------
class SLA_NN(nn.Module):
    def __init__(self, input_dim):
        super(SLA_NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

@st.cache_resource
def load_models():
    models = {}
    try:
        models['LightGBM'] = joblib.load("lightgbm_sla_model.pkl")
    except FileNotFoundError:
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.Booster(model_file="lightgbm_sla_model.txt")
        except Exception:
            models['LightGBM'] = None
    except Exception:
        models['LightGBM'] = None
    try:
        import xgboost as xgb
        xb = xgb.Booster()
        xb.load_model("xgboost_sla_model.json")
        models['XGBoost'] = xb
    except Exception:
        models['XGBoost'] = None
    try:
        from catboost import CatBoostClassifier
        import os
        cb_model_path = "catboost_model.cbm"
        cb_meta_path  = "catboost_meta.json"
        if os.path.exists(cb_model_path) and os.path.exists(cb_meta_path):
            with open(cb_meta_path, "r") as f:
                meta = json.load(f)
            cb = CatBoostClassifier()
            cb.load_model(cb_model_path)
            models['CatBoost'] = {"model": cb, "features": meta.get("feature_cols", []), "tau": float(meta.get("threshold", 0.5)), "temperature": float(meta.get("temperature", 1.0))}
        else:
            models['CatBoost'] = None
    except Exception:
        models['CatBoost'] = None
    try:
        import torch
        tt_path = "tabtransformer_numeric.pt"
        if Path(tt_path).exists():
            ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
            feats = ckpt.get("feature_cols") or ckpt.get("features") or ckpt.get("feat_cols")
            if feats is None:
                raise KeyError("feature_cols not found in checkpoint")
            tau   = float(ckpt.get("threshold", 0.5))
            mean  = np.asarray(ckpt.get("scaler_mean_", ckpt.get("mean")), dtype="float32")
            scale = np.asarray(ckpt.get("scaler_scale_", ckpt.get("scale")), dtype="float32")
            arch  = ckpt.get("arch") or {"d_model":64,"nhead":8,"layers":2,"dim_ff":192,"use_cls":True}
            class NumericTabTransformer(nn.Module):
                def __init__(self, num_features, d_model, nhead, layers, dim_ff, p=0.0, use_cls=True):
                    super().__init__()
                    self.use_cls = use_cls
                    self.scalar_proj = nn.Linear(1, d_model)
                    self.col_embed   = nn.Parameter(torch.randn(num_features, d_model)*0.02)
                    if use_cls:
                        self.cls_token = nn.Parameter(torch.randn(1,1,d_model)*0.02)
                        self.cls_pos   = nn.Parameter(torch.randn(1,1,d_model)*0.02)
                    enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=p, batch_first=True, activation="gelu")
                    self.encoder = nn.TransformerEncoder(enc, layers)
                    self.norm = nn.LayerNorm(d_model)
                    self.head = nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Dropout(p), nn.Linear(d_model,1))
                def forward(self, x):
                    B, F = x.shape
                    x = self.scalar_proj(x.unsqueeze(-1))
                    x = x + self.col_embed.unsqueeze(0).expand(B, -1, -1)
                    if self.use_cls:
                        cls = self.cls_token.expand(B,1,-1) + self.cls_pos
                        x = torch.cat([cls, x], dim=1)
                    z = self.encoder(x)
                    pooled = z[:,0,:] if self.use_cls else z.mean(dim=1)
                    pooled = self.norm(pooled)
                    return self.head(pooled).squeeze(1)
            model = NumericTabTransformer(num_features=len(feats), d_model=arch["d_model"], nhead=arch["nhead"], layers=arch["layers"], dim_ff=arch["dim_ff"], p=0.0, use_cls=arch.get("use_cls", True))
            state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
            if isinstance(state, dict):
                try:
                    model.load_state_dict(state, strict=False)
                except Exception as e:
                    st.warning(f"Loaded TabTransformer but failed to apply state_dict strictly: {e}")
            model.eval()
            models['TabTransformer'] = {"model": model, "features": feats, "tau": tau, "mean": mean, "scale": scale}
        else:
            models['TabTransformer'] = None
    except Exception as e:
        st.warning(f"TabTransformer load error: {e}")
        models['TabTransformer'] = None
    try:
        pt = SLA_NN(input_dim=16)
        pt.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
        pt.eval()
        models['MLP'] = pt
    except Exception:
        models['MLP'] = None
    # GradientBoost Classifier
    try:
        models['GradBoost'] = joblib.load("gradBoost_classifier.pkl")
    except Exception:
        models['GradBoost'] = None
    # Ridge Classifier
    try:
        models['Ridge'] = joblib.load("ridge_classifier.pkl")
    except Exception:
        models['Ridge'] = None
    # SGD Classifier
    try:
        models['SGD'] = joblib.load("sgd_classifier.pkl")
    except Exception:
        models['SGD'] = None
    # TabNet Classifier
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        tn = TabNetClassifier()
        tn.load_model("tabnet_classifier_model.zip")
        models['TabNet'] = tn
    except Exception:
        models['TabNet'] = None
    return models

def predict_sla_violation(model, model_type, features_df):
    try:
        if model is None or model_type == 'Mock':
            np.random.seed(42)
            return np.random.uniform(0, 0.4, len(features_df))
        if model_type == 'LightGBM':
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_df)
                return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
            preds = model.predict(features_df.values)
            return np.asarray(preds).reshape(-1)
        if model_type == 'XGBoost':
            FEATURE_ORDER = [
                "cpu_cores_total","cpu_watts","gpu_cores_total","gpu_vram_total_gb",
                "gpu_watts","ram_total_gb","cpu_cores_used","gpu_cores_used",
                "gpu_vram_used_gb","ram_used_gb","current_cpu_power_w","current_gpu_power_w",
                "req_cpu_cores","req_gpu_cores","req_gpu_vram_gb","req_ram_gb"
            ]
            import xgboost as xgb
            features_df = features_df[FEATURE_ORDER]
            dmatrix = xgb.DMatrix(features_df.values, feature_names=FEATURE_ORDER)
            preds = model.predict(dmatrix)
            return np.asarray(preds).reshape(-1)
        if model_type == 'CatBoost':
            cb = model["model"]
            feats = model["features"]
            T = float(model.get("temperature", 1.0))
            def sigmoid(z): return 1/(1+np.exp(-z))
            def logit(p):
                p = np.clip(p, 1e-6, 1-1e-6)
                return np.log(p/(1-p))
            X = features_df.reindex(columns=feats, fill_value=0.0).values
            p = cb.predict_proba(X)[:, 1]
            return sigmoid(logit(p)/T)
        if model_type == 'TabTransformer':
            import torch
            tt = model["model"]
            feats = model["features"]
            mean = np.asarray(model["mean"], dtype="float32")
            scale = np.asarray(model["scale"], dtype="float32")
            X = features_df.reindex(columns=feats, fill_value=0.0).values.astype("float32")
            X = (X - mean) / scale
            with torch.no_grad():
                logits = tt(torch.as_tensor(X))
                p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            return p
        if model_type == 'MLP':
            with torch.no_grad():
                tensor_input = torch.FloatTensor(features_df.values)
                preds = model(tensor_input).squeeze().numpy()
                return np.asarray(preds).reshape(-1)
        if model_type in ['GradBoost', 'Ridge', 'SGD']:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_df)
                return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
            else:
                preds = model.decision_function(features_df)
                return 1 / (1 + np.exp(-preds))  # sigmoid conversion
        if model_type == 'TabNet':
            import numpy as np
            X = features_df.values
            preds = model.predict_proba(X)[:, 1]
            return np.asarray(preds).reshape(-1)
        return np.random.uniform(0, 0.4, len(features_df))
    except Exception as e:
        st.warning(f"Error in {model_type} prediction: {str(e)}")
        return np.random.uniform(0, 0.4, len(features_df))

# Import the optimized prioritize_machines function
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
    FIXED: Handles single machine and similar values better
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

    # Fit score calculation (improved)
    df["fit_cpu"]  = np.abs(df["remaining_cpu_cores"] - r_cpu) / np.maximum(df["cpu_cores_total"], eps)
    df["fit_gpu"]  = np.abs(df["remaining_gpu_cores"] - r_gpu) / np.maximum(df["gpu_cores_total"], eps)
    df["fit_vram"] = np.abs(df["remaining_gpu_vram_gb"] - r_vram) / np.maximum(df["gpu_vram_total_gb"], eps)
    df["fit_ram"]  = np.abs(df["remaining_ram_gb"] - r_ram) / np.maximum(df["ram_total_gb"], eps)

    # Average normalized deviation across the 4 resources
    df["fit_score_raw"] = (df[["fit_cpu","fit_gpu","fit_vram","fit_ram"]].sum(axis=1)) / 4.0

    # FIXED: Better normalization handling
    min_fit = df["fit_score_raw"].min()
    max_fit = df["fit_score_raw"].max()
    fit_range = max_fit - min_fit
    
    if fit_range < eps:
        # If all machines have similar fit, use raw values directly (smaller is better)
        df["fit_score"] = df["fit_score_raw"]
    else:
        # Normal normalization to [0,1]
        df["fit_score"] = (df["fit_score_raw"] - min_fit) / fit_range

    # Energy estimation (improved with more realistic defaults)
    def derive_row_power(row):
        # CPU per-core with better defaults
        cpu_per_core = cpu_power_per_core if cpu_power_per_core is not None else \
            max(1.0, row["cpu_watts"] / max(row["cpu_cores_total"], 1))
            
        # GPU per-core with better defaults  
        gpu_per_core = gpu_power_per_core if gpu_power_per_core is not None else \
            max(0.1, row["gpu_watts"] / max(row["gpu_cores_total"], 1)) if row["gpu_cores_total"] > 0 else 0
            
        # GPU VRAM per-GB
        gpu_vram_per_gb = gpu_vram_power_per_gb if gpu_vram_power_per_gb is not None else \
            max(0.5, row["gpu_watts"] / max(row["gpu_vram_total_gb"], 1)) if row["gpu_vram_total_gb"] > 0 else 0
            
        # RAM per-GB with realistic default
        if ram_power_per_gb is not None:
            ram_per_gb = ram_power_per_gb
        else:
            # More realistic RAM power estimate (typically 0.3-2 W/GB)
            ram_per_gb = 0.5  # Default 0.5W per GB
        
        # Compute incremental power
        inc_power = (
            cpu_per_core * r_cpu + 
            gpu_per_core * r_gpu + 
            gpu_vram_per_gb * r_vram + 
            ram_per_gb * r_ram
        )
        
        return max(0.1, inc_power)  # Ensure minimum power consumption

    df["est_incremental_power_w"] = df.apply(derive_row_power, axis=1)

    # FIXED: Better energy score normalization
    min_e = df["est_incremental_power_w"].min()
    max_e = df["est_incremental_power_w"].max()
    energy_range = max_e - min_e
    
    if energy_range < 0.01:  # Less strict threshold (0.01W instead of 1e-9)
        # If power consumption is very similar, rank by absolute power (lower is better)
        # Normalize to [0,1] where 0 is lowest power
        if max_e > 0:
            df["energy_score"] = df["est_incremental_power_w"] / max_e
        else:
            df["energy_score"] = 0.0
    else:
        # Normal normalization to [0,1] where 0 is lowest power
        df["energy_score"] = (df["est_incremental_power_w"] - min_e) / energy_range

    # Combine scores
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")

    df["final_score"] = alpha * df["energy_score"] + (1.0 - alpha) * df["fit_score"]

    # Sort ascending (lowest final_score is best)
    ranked = df.sort_values(["final_score", "est_incremental_power_w", "fit_score"]).reset_index(drop=True)

    # return ranked df and best machine
    best_row = ranked.iloc[0] if ranked.shape[0] > 0 else None
    return ranked, best_row

# ------------------------
# Preset machine templates
# ------------------------
PRESETS = {
    "Small VM": {"cpu_cores_total": 4, "gpu_cores_total": 0, "gpu_vram_total_gb": 0, "ram_total_gb": 8,
                 "cpu_watts": 65, "gpu_watts": 0, "cost_per_hour": 0.50},
    "GPU Node": {"cpu_cores_total": 8, "gpu_cores_total": 2048, "gpu_vram_total_gb": 16, "ram_total_gb": 32,
                 "cpu_watts": 95, "gpu_watts": 250, "cost_per_hour": 2.80},
    "Balanced": {"cpu_cores_total": 16, "gpu_cores_total": 1024, "gpu_vram_total_gb": 8, "ram_total_gb": 64,
                 "cpu_watts": 120, "gpu_watts": 150, "cost_per_hour": 1.60},
    "High RAM": {"cpu_cores_total": 8, "gpu_cores_total": 0, "gpu_vram_total_gb": 0, "ram_total_gb": 128,
                 "cpu_watts": 85, "gpu_watts": 0, "cost_per_hour": 1.20},
    "Beast": {"cpu_cores_total": 32, "gpu_cores_total": 4096, "gpu_vram_total_gb": 48, "ram_total_gb": 256,
              "cpu_watts": 200, "gpu_watts": 400, "cost_per_hour": 5.50},
}

# ------------------------
# Machine class
# ------------------------
class Machine:
    def __init__(self, specs, name):
        self.name = name
        self.specs = specs.copy()
        self.free_resources = specs.copy()
        self.tasks = []
        self.utilization_history = []
        self.energy_consumed = 0
        self.total_cost = 0
        self.sla_violations = 0

    def update_status(self, current_time):
        finished_tasks = []
        for task_data in self.tasks[:]:
            task_id, task, start_time, finish_time = task_data
            if finish_time <= current_time:
                self.release_task(task)
                finished_tasks.append(task_data)
                self.tasks.remove(task_data)
        
        cpu_util = 1 - (self.free_resources["cpu_cores_total"] / self.specs["cpu_cores_total"])
        gpu_util = 1 - (self.free_resources["gpu_cores_total"] / max(1, self.specs["gpu_cores_total"]))
        ram_util = 1 - (self.free_resources["ram_total_gb"] / self.specs["ram_total_gb"])
        
        self.utilization_history.append({
            'time': current_time,
            'cpu_util': cpu_util,
            'gpu_util': gpu_util,
            'ram_util': ram_util,
            'active_tasks': len(self.tasks)
        })
        
        return finished_tasks

    def can_accommodate(self, task):
        return (self.free_resources["cpu_cores_total"] >= task.get("req_cpu_cores", 0) and
                self.free_resources["gpu_cores_total"] >= task.get("req_gpu_cores", 0) and
                self.free_resources["gpu_vram_total_gb"] >= task.get("req_gpu_vram_gb", 0) and
                self.free_resources["ram_total_gb"] >= task.get("req_ram_gb", 0))

    def is_free(self, current_time):
        self.update_status(current_time)
        return len(self.tasks) == 0

    def allocate_task(self, task, current_time, task_id):
        finish_time = current_time + task.get("execution_time", 1)
        self.tasks.append((task_id, task, current_time, finish_time))
        
        self.free_resources["cpu_cores_total"] -= task.get("req_cpu_cores", 0)
        self.free_resources["gpu_cores_total"] -= task.get("req_gpu_cores", 0)
        self.free_resources["gpu_vram_total_gb"] -= task.get("req_gpu_vram_gb", 0)
        self.free_resources["ram_total_gb"] -= task.get("req_ram_gb", 0)
        
        duration_hours = task.get("execution_time", 1) / 3600
        self.total_cost += self.specs.get("cost_per_hour", 0) * duration_hours
        
        cpu_power = (task.get("req_cpu_cores", 0) / self.specs["cpu_cores_total"]) * self.specs["cpu_watts"]
        gpu_power = (task.get("req_gpu_cores", 0) / max(1, self.specs["gpu_cores_total"])) * self.specs["gpu_watts"]
        self.energy_consumed += (cpu_power + gpu_power) * duration_hours

    def release_task(self, task):
        self.free_resources["cpu_cores_total"] += task.get("req_cpu_cores", 0)
        self.free_resources["gpu_cores_total"] += task.get("req_gpu_cores", 0)
        self.free_resources["gpu_vram_total_gb"] += task.get("req_gpu_vram_gb", 0)
        self.free_resources["ram_total_gb"] += task.get("req_ram_gb", 0)

    def to_dataframe_row(self):
        return {
            "machine_name": self.name,
            "cpu_cores_total": self.specs["cpu_cores_total"],
            "cpu_watts": self.specs["cpu_watts"],
            "gpu_cores_total": self.specs["gpu_cores_total"],
            "gpu_vram_total_gb": self.specs["gpu_vram_total_gb"],
            "gpu_watts": self.specs["gpu_watts"],
            "ram_total_gb": self.specs["ram_total_gb"],
            "cpu_cores_used": self.specs["cpu_cores_total"] - self.free_resources["cpu_cores_total"],
            "gpu_cores_used": self.specs["gpu_cores_total"] - self.free_resources["gpu_cores_total"],
            "gpu_vram_used_gb": self.specs["gpu_vram_total_gb"] - self.free_resources["gpu_vram_total_gb"],
            "ram_used_gb": self.specs["ram_total_gb"] - self.free_resources["ram_total_gb"],
            "current_cpu_power_w": 0.0,
            "current_gpu_power_w": 0.0,
            "cost_per_hour": self.specs.get("cost_per_hour", 0),
            "energy_consumed": self.energy_consumed,
            "total_cost": self.total_cost,
            "sla_violations": self.sla_violations
        }

# ------------------------
# Dashboard Functions
# ------------------------
def create_machine_dashboard():
    if not st.session_state.machines:
        return None
    
    machine_data = []
    for machine in st.session_state.machines:
        if machine.utilization_history:
            latest = machine.utilization_history[-1]
            machine_data.append({
                'Machine': machine.name,
                'CPU': latest['cpu_util'] * 100,
                'GPU': latest['gpu_util'] * 100,
                'RAM': latest['ram_util'] * 100,
                'Tasks': latest['active_tasks']
            })
    
    if not machine_data:
        return None
    
    df = pd.DataFrame(machine_data)
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Resource Utilization (%)', 'Active Tasks'), vertical_spacing=0.15)
    fig.add_trace(go.Bar(x=df['Machine'], y=df['CPU'], name='CPU', marker_color='#FF6B6B'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Machine'], y=df['GPU'], name='GPU', marker_color='#4ECDC4'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Machine'], y=df['RAM'], name='RAM', marker_color='#45B7D1'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Machine'], y=df['Tasks'], name='Tasks', marker_color='#96CEB4', showlegend=False), row=2, col=1)
    fig.update_layout(height=500, title_text="Real-Time Machine Status")
    fig.update_xaxes(title_text="Machines", row=2, col=1)
    fig.update_yaxes(title_text="Utilization %", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return fig

def create_utilization_timeline():
    if not st.session_state.machines:
        return None
    
    timeline_data = []
    for machine in st.session_state.machines:
        for point in machine.utilization_history:
            timeline_data.extend([
                {'Time': point['time'], 'Machine': machine.name, 'Resource': 'CPU', 'Utilization': point['cpu_util'] * 100},
                {'Time': point['time'], 'Machine': machine.name, 'Resource': 'RAM', 'Utilization': point['ram_util'] * 100},
                {'Time': point['time'], 'Machine': machine.name, 'Resource': 'GPU', 'Utilization': point['gpu_util'] * 100}
            ])
    
    if not timeline_data:
        return None
    
    df = pd.DataFrame(timeline_data)
    fig = px.line(df, x='Time', y='Utilization', color='Resource', facet_col='Machine', facet_col_wrap=2, title="Resource Utilization Timeline")
    fig.update_yaxes(range=[0, 100])
    return fig

def create_task_distribution():
    if not st.session_state.simulation_results:
        return None
    
    results_df = pd.DataFrame(st.session_state.simulation_results)
    successful_results = results_df[results_df['status'] == '✅ Scheduled']
    
    if successful_results.empty:
        return None
    
    machine_counts = successful_results['machine'].value_counts()
    fig = px.pie(values=machine_counts.values, names=machine_counts.index, title="Task Distribution Across Machines")
    return fig

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="VM Placement Simulator", layout="wide")
st.title("🖥️ Real-Time VM Placement Simulator")
st.markdown("*Machine Learning-driven SLA prediction with multiple model support*")

# Load all models
models = load_models()

# Initialize session state
if "machines" not in st.session_state:
    st.session_state.machines = []
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = []
if "current_time" not in st.session_state:
    st.session_state.current_time = 0
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "chart_update_counter" not in st.session_state:
    st.session_state.chart_update_counter = 0

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    available_models = [name for name, model in models.items() if model is not None]
    selected_model = st.radio("Select SLA Prediction Model", available_models, index=0)
    
    # Show model status
    st.subheader("Model Status")
    for name, model in models.items():
        status = "✅ Loaded" if model is not None else "❌ Not Available"
        st.text(f"{name}: {status}")
    
    alpha = st.slider("Energy vs Fit Weight (α)", 0.0, 1.0, 0.5, 0.1)
    simulation_speed = st.selectbox("Simulation Speed", [0.5, 1, 2, 5], index=1)
    task_interval = st.slider("Time Between Tasks (seconds)", 1, 20, 5, 1)
    
    st.info("📋 **Workflow:**\n1. Check if machine is free\n2. Check if machine can accommodate task\n3. Use selected ML model for SLA prediction\n4. Apply optimization to select best machine from safe machines")

# Real-time Dashboard (Always visible)
st.header("📊 Live Machine Dashboard")

dashboard_col1, dashboard_col2 = st.columns(2)

# Use placeholders to prevent duplicate keys
with dashboard_col1:
    machine_chart_placeholder = st.empty()
with dashboard_col2:
    task_chart_placeholder = st.empty()

timeline_placeholder = st.empty()

metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
with metrics_col1:
    total_machines_metric = st.empty()
with metrics_col2:
    busy_machines_metric = st.empty()
with metrics_col3:
    successful_tasks_metric = st.empty()
with metrics_col4:
    total_energy_metric = st.empty()

def update_dashboard():
    """Update dashboard with unique keys to prevent ID collisions"""
    
    # Increment counter for unique keys
    st.session_state.chart_update_counter += 1
    counter = st.session_state.chart_update_counter
    
    # Update machine dashboard
    machine_fig = create_machine_dashboard()
    if machine_fig:
        machine_chart_placeholder.plotly_chart(machine_fig, use_container_width=True, key=f"machine_chart_{counter}")
    else:
        machine_chart_placeholder.info("No machine data available")
    
    # Update task distribution
    task_fig = create_task_distribution()
    if task_fig:
        task_chart_placeholder.plotly_chart(task_fig, use_container_width=True, key=f"task_chart_{counter}")
    else:
        task_chart_placeholder.info("No task assignments yet")
    
    # Update timeline
    timeline_fig = create_utilization_timeline()
    if timeline_fig:
        timeline_placeholder.plotly_chart(timeline_fig, use_container_width=True, key=f"timeline_chart_{counter}")
    else:
        timeline_placeholder.info("No utilization timeline data")
    
    # Update metrics
    total_machines_metric.metric("Total Machines", len(st.session_state.machines))
    busy_count = sum(1 for m in st.session_state.machines if len(m.tasks) > 0)
    busy_machines_metric.metric("Busy Machines", busy_count)
    successful_count = len([r for r in st.session_state.simulation_results if r.get('status') == '✅ Scheduled'])
    successful_tasks_metric.metric("Successful Tasks", successful_count)
    total_energy = sum(m.energy_consumed for m in st.session_state.machines)
    total_energy_metric.metric("Total Energy", f"{total_energy:.2f} kWh")

# Initial dashboard update
update_dashboard()

# Machine Management
st.header("🔧 Machine Pool Management")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Add Preset Machine")
    preset = st.selectbox("Choose preset", list(PRESETS.keys()))
    
    col_preset1, col_preset2 = st.columns(2)
    with col_preset1:
        if st.button("➕ Add Preset Machine", use_container_width=True):
            idx = len(st.session_state.machines) + 1
            st.session_state.machines.append(Machine(PRESETS[preset], f"{preset}-{idx}"))
            st.success(f"Added {preset}-{idx}")
            update_dashboard()
    
    with col_preset2:
        if st.button("🗑️ Clear All Machines", use_container_width=True):
            st.session_state.machines = []
            st.session_state.simulation_results = []
            st.success("Cleared all machines")
            update_dashboard()

    st.subheader("Add Custom Machine")
    with st.form("custom_form"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cpu = st.number_input("CPU cores", 1, 128, 8)
            gpu = st.number_input("GPU cores", 0, 8192, 0)
            vram = st.number_input("GPU VRAM (GB)", 0, 128, 0)
        with col_c2:
            ram = st.number_input("RAM (GB)", 1, 512, 16)
            cpu_w = st.number_input("CPU Watts", 1, 400, 95)
            gpu_w = st.number_input("GPU Watts", 0, 600, 0)
            cost = st.number_input("Cost per hour ($)", 0.1, 10.0, 1.0)
        
        if st.form_submit_button("➕ Add Custom Machine", use_container_width=True):
            spec = {"cpu_cores_total": cpu, "gpu_cores_total": gpu, "gpu_vram_total_gb": vram,
                   "ram_total_gb": ram, "cpu_watts": cpu_w, "gpu_watts": gpu_w, "cost_per_hour": cost}
            idx = len(st.session_state.machines) + 1
            st.session_state.machines.append(Machine(spec, f"Custom-{idx}"))
            st.success(f"Added Custom-{idx}")
            update_dashboard()

with col2:
    st.subheader("Current Machine Pool")
    if st.session_state.machines:
        machine_data = []
        for m in st.session_state.machines:
            data = m.to_dataframe_row()
            data["utilization_cpu"] = (data["cpu_cores_used"] / data["cpu_cores_total"]) * 100
            data["utilization_ram"] = (data["ram_used_gb"] / data["ram_total_gb"]) * 100
            data["active_tasks"] = len(m.tasks)
            machine_data.append(data)
        
        machines_df = pd.DataFrame(machine_data)
        st.metric("Total Machines", len(st.session_state.machines))
        display_cols = ["machine_name", "cpu_cores_total", "ram_total_gb", "utilization_cpu", "utilization_ram", "active_tasks"]
        st.dataframe(machines_df[display_cols], use_container_width=True)
    else:
        st.info("No machines added yet. Add some machines to get started!")

# Workload Management - FIXED VERSION
st.header("📋 Workload Management")
col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.subheader("Upload Workload CSV")
    workload_file = st.file_uploader("Choose CSV file", type="csv")
    
    if workload_file:
        try:
            uploaded_df = pd.read_csv(workload_file)
            st.dataframe(uploaded_df.head(), use_container_width=True)
            
            # Define task-specific columns (what we expect from the workload)
            TASK_COLUMNS = [
                'req_cpu_cores', 'req_gpu_cores', 'req_gpu_vram_gb', 'req_ram_gb', 'execution_time'
            ]
            
            # Check which required task columns are present
            available_task_cols = [col for col in TASK_COLUMNS if col in uploaded_df.columns]
            missing_task_cols = [col for col in TASK_COLUMNS if col not in uploaded_df.columns]
            
            if len(available_task_cols) >= 4:  # At least CPU, RAM, and execution time required
                # Extract only task-specific columns and clean the data
                task_df = uploaded_df[available_task_cols].copy()
                
                # Fill missing values with defaults
                for col in TASK_COLUMNS:
                    if col not in task_df.columns:
                        if col == 'execution_time':
                            task_df[col] = 30.0  # Default 30 seconds
                        else:
                            task_df[col] = 0  # Default 0 for resource requirements
                
                # Ensure all values are numeric and positive
                for col in TASK_COLUMNS:
                    task_df[col] = pd.to_numeric(task_df[col], errors='coerce').fillna(0)
                    task_df[col] = task_df[col].abs()  # Ensure positive values
                
                # Store in session state
                st.session_state['tasksdf'] = task_df
                
                st.success(f"✅ Successfully loaded {len(task_df)} tasks!")
                
                # Show summary
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Total Tasks", len(task_df))
                with col_s2:
                    avg_cpu = task_df["req_cpu_cores"].mean()
                    st.metric("Avg CPU Req", f"{avg_cpu:.1f}")
                with col_s3:
                    avg_ram = task_df["req_ram_gb"].mean()
                    st.metric("Avg RAM Req", f"{avg_ram:.1f} GB")
                
                # Show what columns were used
                st.info(f"📋 **Task columns used**: {', '.join(available_task_cols)}")
                if missing_task_cols:
                    st.warning(f"⚠️ **Missing columns** (using defaults): {', '.join(missing_task_cols)}")
                
                # Show preview of processed data
                st.subheader("Processed Task Data")
                st.dataframe(task_df.head(), use_container_width=True)
                
            else:
                st.error(f"❌ **Invalid CSV format**\n\nRequired columns: {', '.join(TASK_COLUMNS[:4])}\n\nFound: {', '.join(available_task_cols) if available_task_cols else 'None'}")
                st.info("💡 **CSV should contain task requirements only**. Machine data comes from the Machine Pool.")
                
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
            st.info("💡 Make sure your CSV has columns like: req_cpu_cores, req_ram_gb, execution_time")

with col_w2:
    st.subheader("Or Generate Sample Workload")
    with st.form("sample_workload_form"):
        num_tasks = st.number_input("Number of tasks", 1, 100, 10)
        task_complexity = st.selectbox("Task complexity", ["Light", "Mixed", "Heavy"])
        
        if st.form_submit_button("Generate Sample Workload"):
            np.random.seed(42)
            
            sample_tasks = []
            for i in range(num_tasks):
                if task_complexity == "Light":
                    cpu_range, ram_range, gpu_range = (1, 4), (1, 8), (0, 0)
                elif task_complexity == "Mixed":
                    cpu_range, ram_range, gpu_range = (1, 8), (2, 16), (0, 512)
                else:  # Heavy
                    cpu_range, ram_range, gpu_range = (4, 16), (8, 64), (0, 2048)

                # Generate ONLY task-specific data (requirements)
                task = {
                    "req_cpu_cores": np.random.randint(cpu_range[0], cpu_range[1] + 1),
                    "req_gpu_cores": np.random.randint(gpu_range[0], gpu_range[1] + 1),
                    "req_gpu_vram_gb": np.random.randint(0, 16) if gpu_range[1] > 0 else 0,
                    "req_ram_gb": np.random.randint(ram_range[0], ram_range[1] + 1),
                    "execution_time": round(np.random.uniform(5, 60), 2)
                }
                sample_tasks.append(task)
            
            tasks_df = pd.DataFrame(sample_tasks)
            st.session_state['tasksdf'] = tasks_df
            st.success(f"Generated {num_tasks} sample tasks")
            st.dataframe(tasks_df, use_container_width=True)

# Real-time Simulation - UPDATED VERSION
st.session_state.simulation_results = []
st.session_state.current_time = 0
st.session_state.simulation_running = False
for machine in st.session_state.machines:
    machine.tasks = []
    machine.free_resources = machine.specs.copy()
    machine.utilization_history = []
    machine.energy_consumed = 0
    machine.total_cost = 0
    machine.sla_violations = 0
    
st.header("🚀 Real-Time Simulation")

# Check conditions and provide helpful messages
has_tasks = 'tasksdf' in st.session_state and len(st.session_state.get('tasksdf', [])) > 0
has_machines = len(st.session_state.machines) > 0

# Show status
col_status1, col_status2 = st.columns(2)
with col_status1:
    if has_tasks:
        st.success(f"✅ Tasks ready: {len(st.session_state['tasksdf'])} tasks loaded")
    else:
        st.warning("⚠️ No tasks loaded. Upload CSV or generate sample workload above.")

with col_status2:
    if has_machines:
        st.success(f"✅ Machines ready: {len(st.session_state.machines)} machines available")
    else:
        st.warning("⚠️ No machines available. Add machines in the Machine Pool section above.")

# Always show the simulation controls, but disable the button if conditions aren't met
col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    # Determine if button should be disabled
    button_disabled = st.session_state.simulation_running or not has_tasks or not has_machines
    
    # Create the button with appropriate state
    if st.button("▶️ Start Real-Time Simulation", 
                use_container_width=True, 
                type="primary", 
                disabled=button_disabled):
        
        if not has_tasks:
            st.error("❌ Cannot start simulation: No tasks available. Please upload a CSV or generate sample tasks.")
        elif not has_machines:
            st.error("❌ Cannot start simulation: No machines available. Please add machines to the pool.")
        else:
            # Start simulation with proper data separation
            st.session_state.simulation_running = True
            st.session_state.simulation_results = []
            
            # Get selected model
            current_model = models[selected_model]
            
            progress_container = st.container()
            live_results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_task_info = st.empty()
            
            with live_results_container:
                st.subheader("Live Task Processing")
                results_table = st.empty()

                for i, task in st.session_state['tasksdf'].iterrows():
                    progress_bar.progress((i + 1) / len(st.session_state['tasksdf']))
                    status_text.text(f"Processing Task {i+1}/{len(st.session_state['tasksdf'])} using {selected_model}")
                    
                    with current_task_info.container():
                        st.info(f"**Task {i+1}**: CPU: {task.get('req_cpu_cores', 0)}, GPU: {task.get('req_gpu_cores', 0)}, RAM: {task.get('req_ram_gb', 0)} GB, Time: {task.get('execution_time', 0)}s")
                    
                    current_time = st.session_state.current_time
                    
                    # Update machine statuses
                    for machine in st.session_state.machines:
                        machine.update_status(current_time)
                    
                    update_dashboard()
                    
                    # Find free machines that can accommodate the task
                    free_machines = [m for m in st.session_state.machines if m.is_free(current_time)]
                    
                    if not free_machines:
                        result = {"task_id": i+1, "status": "❌ No free machines", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A", "model_used": selected_model}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    capable_machines = [m for m in free_machines if m.can_accommodate(task)]
                    
                    if not capable_machines:
                        result = {"task_id": i+1, "status": "❌ No suitable free machine", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A", "model_used": selected_model}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    # SLA prediction using machine data + task requirements
                    sla_safe_machines = []
                    sla_predictions = {}
                    
                    for machine in capable_machines:
                        # CRITICAL: Combine machine specs with task requirements for SLA prediction
                        features = pd.DataFrame([{
                            # Machine specifications (from machine pool)
                            "cpu_cores_total": machine.specs["cpu_cores_total"],
                            "cpu_watts": machine.specs["cpu_watts"],
                            "gpu_cores_total": machine.specs["gpu_cores_total"],
                            "gpu_vram_total_gb": machine.specs["gpu_vram_total_gb"],
                            "gpu_watts": machine.specs["gpu_watts"],
                            "ram_total_gb": machine.specs["ram_total_gb"],
                            # Current machine usage (from machine state)
                            "cpu_cores_used": machine.specs["cpu_cores_total"] - machine.free_resources["cpu_cores_total"],
                            "gpu_cores_used": machine.specs["gpu_cores_total"] - machine.free_resources["gpu_cores_total"],
                            "gpu_vram_used_gb": machine.specs["gpu_vram_total_gb"] - machine.free_resources["gpu_vram_total_gb"],
                            "ram_used_gb": machine.specs["ram_total_gb"] - machine.free_resources["ram_total_gb"],
                            "current_cpu_power_w": 50.0,  # Estimated current power
                            "current_gpu_power_w": 100.0, # Estimated current power
                            # Task requirements (from task dataset)
                            "req_cpu_cores": task.get("req_cpu_cores", 0),
                            "req_gpu_cores": task.get("req_gpu_cores", 0),
                            "req_gpu_vram_gb": task.get("req_gpu_vram_gb", 0),
                            "req_ram_gb": task.get("req_ram_gb", 0),
                        }])
                        
                        try:
                            prediction = predict_sla_violation(current_model, selected_model, features)[0]
                            sla_predictions[machine.name] = prediction
                            
                            # Only include machines where model predicts NO SLA violation (< 0.5)
                            if prediction < 0.5:
                                sla_safe_machines.append(machine)
                        except Exception as e:
                            sla_predictions[machine.name] = 0.0
                            sla_safe_machines.append(machine)
                    
                    if not sla_safe_machines:
                        result = {"task_id": i+1, "status": "⚠️ All machines predict SLA violation", "machine": None, "sla_prediction": "High", "energy_score": "N/A", "fit_score": "N/A", "model_used": selected_model}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    # Apply optimization to select best machine
                    safe_machines_df = pd.DataFrame([m.to_dataframe_row() for m in sla_safe_machines])
                    task_reqs = {
                        "req_cpu_cores": task.get("req_cpu_cores", 0),
                        "req_gpu_cores": task.get("req_gpu_cores", 0),
                        "req_gpu_vram_gb": task.get("req_gpu_vram_gb", 0),
                        "req_ram_gb": task.get("req_ram_gb", 0)
                    }
                    
                    try:
                        ranked_machines, best_machine_row = prioritize_machines(
                            safe_machines_df, 
                            task_reqs, 
                            alpha=alpha
                        )
                        
                        if best_machine_row is not None:
                            chosen_machine = next(m for m in sla_safe_machines if m.name == best_machine_row["machine_name"])
                            chosen_machine.allocate_task(task, current_time, f"task_{i+1}")
                            
                            energy_score = best_machine_row.get('energy_score', 0)
                            fit_score = best_machine_row.get('fit_score', 0)
                            
                            result = {
                                "task_id": i+1,
                                "status": "✅ Scheduled",
                                "machine": chosen_machine.name,
                                "sla_prediction": f"{sla_predictions[chosen_machine.name]:.3f}",
                                "energy_score": f"{energy_score:.3f}",
                                "fit_score": f"{fit_score:.3f}",
                                "model_used": selected_model
                            }
                        else:
                            result = {"task_id": i+1, "status": "❌ Optimization failed", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A", "model_used": selected_model}
                    except Exception as e:
                        result = {"task_id": i+1, "status": f"❌ Error: {str(e)[:50]}", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A", "model_used": selected_model}
                    
                    st.session_state.simulation_results.append(result)
                    results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                    results_table.dataframe(results_df_temp, use_container_width=True)
                    update_dashboard()
                    st.session_state.current_time += task_interval
                    time.sleep(1.0 / simulation_speed)
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Real-time simulation completed using {selected_model}!")
                current_task_info.success("All tasks processed!")
                
            st.session_state.simulation_running = False

    # Show button status
    if button_disabled and not st.session_state.simulation_running:
        if not has_tasks and not has_machines:
            st.caption("⚠️ Button disabled: Need both tasks and machines")
        elif not has_tasks:
            st.caption("⚠️ Button disabled: Need tasks (upload CSV or generate sample)")
        elif not has_machines:
            st.caption("⚠️ Button disabled: Need machines (add to machine pool)")

with col_sim2:
    if st.button("⏹️ Reset Simulation", use_container_width=True):
        st.session_state.simulation_results = []
        st.session_state.current_time = 0
        st.session_state.simulation_running = False
        for machine in st.session_state.machines:
            machine.tasks = []
            machine.free_resources = machine.specs.copy()
            machine.utilization_history = []
            machine.energy_consumed = 0
            machine.total_cost = 0
            machine.sla_violations = 0
        st.success("Simulation reset successfully!")
        update_dashboard()

# Live Results Section
if st.session_state.simulation_results:
    st.header("📈 Live Simulation Results")
    
    results_df = pd.DataFrame(st.session_state.simulation_results)
    successful_tasks = len(results_df[results_df['status'] == '✅ Scheduled'])
    failed_tasks = len(results_df) - successful_tasks
    success_rate = (successful_tasks / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    col_lr1, col_lr2, col_lr3, col_lr4 = st.columns(4)
    
    with col_lr1:
        st.metric("Tasks Processed", len(results_df))
    with col_lr2:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col_lr3:
        total_cost = sum(m.total_cost for m in st.session_state.machines)
        st.metric("Total Cost", f"${total_cost:.2f}")
    with col_lr4:
        total_energy = sum(m.energy_consumed for m in st.session_state.machines)
        st.metric("Total Energy", f"{total_energy:.2f} kWh")
    
    st.subheader("Task Processing Log")
    st.dataframe(results_df.tail(10), use_container_width=True)

# Advanced Analytics
if st.session_state.simulation_results and not st.session_state.simulation_running:
    st.header("📊 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Machine Efficiency", "Model Comparison"])
    
    with tab1:
        results_df = pd.DataFrame(st.session_state.simulation_results)
        successful_results = results_df[results_df['status'] == '✅ Scheduled']
        
        if not successful_results.empty:
            try:
                sla_predictions_numeric = pd.to_numeric(successful_results['sla_prediction'], errors='coerce').dropna()
                energy_scores = pd.to_numeric(successful_results['energy_score'], errors='coerce').dropna()
                fit_scores = pd.to_numeric(successful_results['fit_score'], errors='coerce').dropna()
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    if len(sla_predictions_numeric) > 0:
                        fig_sla_hist = px.histogram(x=sla_predictions_numeric, nbins=20, title="SLA Prediction Distribution", labels={'x': 'SLA Prediction Score', 'y': 'Count'})
                        st.plotly_chart(fig_sla_hist, use_container_width=True)
                
                with col_p2:
                    if len(energy_scores) > 0 and len(fit_scores) > 0:
                        fig_scatter = px.scatter(x=energy_scores, y=fit_scores, title="Energy vs Fit Scores", 
                                               labels={'x': 'Energy Score', 'y': 'Fit Score'})
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Display metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    if len(sla_predictions_numeric) > 0:
                        avg_sla_prediction = sla_predictions_numeric.mean()
                        st.metric("Avg SLA Prediction", f"{avg_sla_prediction:.3f}")
                        high_risk_count = len(sla_predictions_numeric[sla_predictions_numeric > 0.5])
                        st.metric("High Risk Tasks", high_risk_count)
                with col_m2:
                    if len(energy_scores) > 0:
                        avg_energy = energy_scores.mean()
                        st.metric("Avg Energy Score", f"{avg_energy:.3f}")
                with col_m3:
                    if len(fit_scores) > 0:
                        avg_fit = fit_scores.mean()
                        st.metric("Avg Fit Score", f"{avg_fit:.3f}")
                        
            except:
                st.info("Analysis data not available yet")
        else:
            st.info("No successful tasks to analyze")
    
    with tab2:
        if st.session_state.machines:
            # Time series utilization chart
            timeline_data = []
            for machine in st.session_state.machines:
                for point in machine.utilization_history:
                    timeline_data.extend([
                        {'Time': point['time'], 'Machine': machine.name, 'Resource': 'CPU', 'Utilization': point['cpu_util'] * 100},
                        {'Time': point['time'], 'Machine': machine.name, 'Resource': 'RAM', 'Utilization': point['ram_util'] * 100},
                        {'Time': point['time'], 'Machine': machine.name, 'Resource': 'GPU', 'Utilization': point['gpu_util'] * 100}
                    ])
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                # Time series utilization chart
                fig_timeseries = px.line(
                    timeline_df,
                    x='Time',
                    y='Utilization',
                    color='Resource',
                    facet_col='Machine',
                    facet_col_wrap=2,
                    title="Resource Utilization Over Time"
                )
                fig_timeseries.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_timeseries, use_container_width=True)
                
                # Current utilization snapshot
                latest_util_data = []
                for machine in st.session_state.machines:
                    if machine.utilization_history:
                        latest_util = machine.utilization_history[-1]
                        latest_util_data.append({
                            'Machine': machine.name,
                            'CPU Utilization': latest_util['cpu_util'] * 100,
                            'GPU Utilization': latest_util['gpu_util'] * 100,
                            'RAM Utilization': latest_util['ram_util'] * 100,
                            'Active Tasks': latest_util['active_tasks']
                        })
                
                if latest_util_data:
                    latest_df = pd.DataFrame(latest_util_data)
                    
                    # Current utilization bar chart
                    fig_current = px.bar(
                        latest_df,
                        x='Machine',
                        y=['CPU Utilization', 'GPU Utilization', 'RAM Utilization'],
                        title="Current Resource Utilization by Machine",
                        barmode='group'
                    )
                    st.plotly_chart(fig_current, use_container_width=True)
                    
                    # Active tasks distribution
                    fig_tasks = px.bar(
                        latest_df,
                        x='Machine',
                        y='Active Tasks',
                        title="Current Active Tasks per Machine"
                    )
                    st.plotly_chart(fig_tasks, use_container_width=True)
            else:
                st.info("No utilization data available yet.")
        else:
            st.info("No machines to display utilization for.")
    
    with tab3:
        if st.session_state.simulation_results:
            results_df = pd.DataFrame(st.session_state.simulation_results)
            
            if 'model_used' in results_df.columns:
                # Model performance comparison
                model_stats = results_df.groupby('model_used').agg({
                    'status': lambda x: (x == '✅ Scheduled').sum(),
                    'task_id': 'count'
                }).reset_index()
                model_stats.columns = ['Model', 'Successful_Tasks', 'Total_Tasks']
                model_stats['Success_Rate'] = (model_stats['Successful_Tasks'] / model_stats['Total_Tasks']) * 100
                
                col_mc1, col_mc2 = st.columns(2)
                
                with col_mc1:
                    fig_model_performance = px.bar(
                        model_stats,
                        x='Model',
                        y='Success_Rate',
                        title="Model Success Rate Comparison"
                    )
                    st.plotly_chart(fig_model_performance, use_container_width=True)
                
                with col_mc2:
                    fig_model_tasks = px.bar(
                        model_stats,
                        x='Model',
                        y=['Successful_Tasks', 'Total_Tasks'],
                        title="Tasks Processed by Model",
                        barmode='group'
                    )
                    st.plotly_chart(fig_model_tasks, use_container_width=True)
                
                st.subheader("Model Performance Summary")
                st.dataframe(model_stats, use_container_width=True)
                
                # SLA prediction distribution by model
                successful_df = results_df[results_df['status'] == '✅ Scheduled'].copy()
                if not successful_df.empty and 'sla_prediction' in successful_df.columns:
                    successful_df['sla_prediction_numeric'] = pd.to_numeric(successful_df['sla_prediction'], errors='coerce')
                    successful_df = successful_df.dropna(subset=['sla_prediction_numeric'])
                    
                    if not successful_df.empty:
                        fig_sla_by_model = px.box(
                            successful_df,
                            x='model_used',
                            y='sla_prediction_numeric',
                            title="SLA Prediction Distribution by Model"
                        )
                        st.plotly_chart(fig_sla_by_model, use_container_width=True)
            else:
                st.info("No model comparison data available yet.")

# Export functionality
if st.session_state.simulation_results:
    st.header("💾 Export Data")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        if st.button("📊 Export Results"):
            results_df = pd.DataFrame(st.session_state.simulation_results)
            csv = results_df.to_csv(index=False)
            st.download_button("Download Results CSV", csv, f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    
    with col_ex2:
        if st.button("🖥️ Export Machine Data"):
            machine_data = [m.to_dataframe_row() for m in st.session_state.machines]
            machines_df = pd.DataFrame(machine_data)
            csv = machines_df.to_csv(index=False)
            st.download_button("Download Machine Data CSV", csv, f"machine_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    
    with col_ex3:
        if st.button("📈 Export Utilization History"):
            util_data = []
            for machine in st.session_state.machines:
                for point in machine.utilization_history:
                    util_data.append({
                        'machine': machine.name,
                        'time': point['time'],
                        'cpu_util': point['cpu_util'],
                        'gpu_util': point['gpu_util'],
                        'ram_util': point['ram_util'],
                        'active_tasks': point['active_tasks']
                    })
            
            if util_data:
                util_df = pd.DataFrame(util_data)
                csv = util_df.to_csv(index=False)
                st.download_button("Download Utilization CSV", csv, f"utilization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
### 🚀 Multi-Model Features:
- **4 Model Support**: LightGBM, XGBoost, PyTorch Neural Network, and Mock model
- **Model Selection**: Radio button to choose which model to use for SLA prediction
- **Model Comparison**: Analytics tab showing performance comparison between models
- **Unified Prediction**: Single interface handling different model types seamlessly

### 🔧 Technical Implementation:
- **Binary SLA Classification**: All models predict 0-1 probability of SLA violation
- **Multi-Objective Optimization**: Balances energy efficiency with resource fit
- **Real-Time Processing**: Live updates during simulation with selected model
- **Model Status**: Shows which models are loaded and available

### ⚡ Performance Features:
- **Model-Specific Analytics**: Compare success rates and prediction distributions
- **Flexible Model Switching**: Change models between simulation runs
- **Error Handling**: Graceful fallbacks when models fail to load
- **Consistent Interface**: Same workflow regardless of selected model

### 📊 Analytics:
- **Model Performance Comparison**: Success rates and task processing by model
- **SLA Distribution Analysis**: Box plots showing prediction ranges by model  
- **Real-Time Utilization**: Historical machine usage patterns
- **Export Capabilities**: Download results, machine data, and utilization history
""")