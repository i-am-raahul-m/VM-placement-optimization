import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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

    # Fit score (how tightly the request fits into remaining resources)
    df["fit_cpu"]  = np.abs(df["remaining_cpu_cores"] - r_cpu) / (df["cpu_cores_total"].replace(0, eps))
    df["fit_gpu"]  = np.abs(df["remaining_gpu_cores"] - r_gpu) / (df["gpu_cores_total"].replace(0, eps))
    df["fit_vram"] = np.abs(df["remaining_gpu_vram_gb"] - r_vram) / (df["gpu_vram_total_gb"].replace(0, eps))
    df["fit_ram"]  = np.abs(df["remaining_ram_gb"] - r_ram) / (df["ram_total_gb"].replace(0, eps))

    # average normalized deviation across the 4 resources
    df["fit_score_raw"] = (df[["fit_cpu","fit_gpu","fit_vram","fit_ram"]].sum(axis=1)) / 4.0

    # Normalize fit_score_raw to 0..1 across machines (lower = better)
    min_fit = df["fit_score_raw"].min()
    max_fit = df["fit_score_raw"].max()
    if abs(max_fit - min_fit) < eps:
        df["fit_score"] = 0.0
    else:
        df["fit_score"] = (df["fit_score_raw"] - min_fit) / (max_fit - min_fit)

    # Energy estimation
    def derive_row_power(row):
        # CPU per-core
        cpu_per_core = cpu_power_per_core if cpu_power_per_core is not None else \
            (row["cpu_watts"] / max(row["cpu_cores_total"], eps))
        # GPU per-core
        gpu_per_core = gpu_power_per_core if gpu_power_per_core is not None else \
            (row["gpu_watts"] / max(row["gpu_cores_total"], eps))
        # GPU vram per-GB: approximate as (gpu_watts / gpu_vram_total_gb) if available
        gpu_vram_per_gb = gpu_vram_power_per_gb if gpu_vram_power_per_gb is not None else \
            (row["gpu_watts"] / max(row["gpu_vram_total_gb"], eps))
        # RAM per-GB: if provided use, else estimate with a small default (0.3 W/GB) OR infer from current power proportion
        if ram_power_per_gb is not None:
            ram_per_gb = ram_power_per_gb
        else:
            # heuristic: assume RAM power is small fraction of current CPU+GPU power relative to ram size
            cur_power = float(row.get("current_cpu_power_w", 0.0)) + float(row.get("current_gpu_power_w", 0.0))
            total_ram_gb = max(row["ram_total_gb"], 1.0)
            # pick a small default but scale with observed power:
            ram_per_gb = max(0.05, min(2.0, (cur_power * 0.02) / total_ram_gb))  # between 0.05 and 2 W/GB heuristic
        # compute incremental power
        inc_power = cpu_per_core * r_cpu + gpu_per_core * r_gpu + gpu_vram_per_gb * r_vram + ram_per_gb * r_ram
        return inc_power

    df["est_incremental_power_w"] = df.apply(derive_row_power, axis=1)

    # Normalize energy to 0..1 so we can combine with fit_score
    min_e = df["est_incremental_power_w"].min()
    max_e = df["est_incremental_power_w"].max()
    if abs(max_e - min_e) < eps:
        df["energy_score"] = 0.0
    else:
        df["energy_score"] = (df["est_incremental_power_w"] - min_e) / (max_e - min_e)

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
# Load trained LightGBM model
# ------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("lightgbm_sla_model.pkl")
    except FileNotFoundError:
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file="lightgbm_sla_model.txt")
        except (FileNotFoundError, ImportError):
            st.warning("⚠️ lightgbm_sla_model.txt not found or LightGBM not installed. Using mock predictions.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {str(e)}. Using mock predictions.")
        return None

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
# FIXED: Real-time Dashboard Functions with stable keys
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
st.markdown("*Machine Learning-driven SLA prediction with live monitoring*")

model = load_model()

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

with st.sidebar:
    st.header("⚙️ Configuration")
    alpha = st.slider("Energy vs Fit Weight (α)", 0.0, 1.0, 0.5, 0.1)
    simulation_speed = st.selectbox("Simulation Speed", [0.5, 1, 2, 5], index=1)
    task_interval = st.slider("Time Between Tasks (seconds)", 1, 20, 5, 1)

# Real-time Dashboard (Always visible)
st.header("📊 Live Machine Dashboard")

dashboard_col1, dashboard_col2 = st.columns(2)

# FIXED: Use placeholders to prevent duplicate keys
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
        display_cols = ["machine_name", "cpu_cores_total", "ram_total_gb", "utilization_cpu", "utilization_ram", "active_tasks"]
        st.dataframe(machines_df[display_cols], use_container_width=True)
    else:
        st.info("No machines added yet. Add some machines to get started!")

# Workload Management
st.header("📋 Workload Management")
col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.subheader("Upload Workload CSV")
    workload_file = st.file_uploader("Choose CSV file", type="csv")
    
    if workload_file:
        tasks_df = pd.read_csv(workload_file)
        st.dataframe(tasks_df.head(), use_container_width=True)
        
        if len(tasks_df) > 0:
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Tasks", len(tasks_df))
            with col_s2:
                avg_cpu = tasks_df.get("req_cpu_cores", [0]).mean()
                st.metric("Avg CPU Req", f"{avg_cpu:.1f}")
            with col_s3:
                avg_ram = tasks_df.get("req_ram_gb", [0]).mean()
                st.metric("Avg RAM Req", f"{avg_ram:.1f} GB")

with col_w2:
    st.subheader("Or Generate Sample Workload")
    with st.form("sample_workload_form"):
        num_tasks = st.number_input("Number of tasks", 1, 100, 10)
        task_complexity = st.selectbox("Task complexity", ["Light", "Mixed", "Heavy"])
        
        if st.form_submit_button("Generate Sample Workload"):
            np.random.seed(42)
            
            sample_tasks = []
            for i in range(num_tasks):
                # Generate machine specifications (similar to presets but with variation)
                machine_types = ["small", "balanced", "gpu", "high_ram", "beast"]
                machine_type = np.random.choice(machine_types)
                
                if machine_type == "small":
                    cpu_total = np.random.randint(4, 8)
                    gpu_total = np.random.randint(0, 512)
                    ram_total = np.random.randint(8, 16)
                    cpu_watts = np.random.randint(65, 85)
                    gpu_watts = np.random.randint(0, 100)
                elif machine_type == "balanced":
                    cpu_total = np.random.randint(12, 20)
                    gpu_total = np.random.randint(512, 1536)
                    ram_total = np.random.randint(32, 96)
                    cpu_watts = np.random.randint(100, 140)
                    gpu_watts = np.random.randint(120, 180)
                elif machine_type == "gpu":
                    cpu_total = np.random.randint(6, 12)
                    gpu_total = np.random.randint(1536, 3072)
                    ram_total = np.random.randint(16, 48)
                    cpu_watts = np.random.randint(80, 110)
                    gpu_watts = np.random.randint(200, 300)
                elif machine_type == "high_ram":
                    cpu_total = np.random.randint(6, 12)
                    gpu_total = np.random.randint(0, 512)
                    ram_total = np.random.randint(64, 256)
                    cpu_watts = np.random.randint(70, 100)
                    gpu_watts = np.random.randint(0, 50)
                else:  # beast
                    cpu_total = np.random.randint(24, 40)
                    gpu_total = np.random.randint(3072, 8192)
                    ram_total = np.random.randint(128, 512)
                    cpu_watts = np.random.randint(180, 220)
                    gpu_watts = np.random.randint(300, 450)
                
                # Generate current usage (what's already being used)
                cpu_used = np.random.randint(0, max(1, cpu_total // 2))
                gpu_used = np.random.randint(0, max(1, gpu_total // 2)) if gpu_total > 0 else 0
                vram_total = np.random.randint(4, 48) if gpu_total > 0 else 0
                vram_used = np.random.uniform(0, vram_total * 0.6) if vram_total > 0 else 0
                ram_used = np.random.uniform(0, ram_total * 0.7)
                
                # Generate current power consumption based on usage
                cpu_utilization = cpu_used / max(1, cpu_total)
                gpu_utilization = gpu_used / max(1, gpu_total) if gpu_total > 0 else 0
                current_cpu_power = cpu_watts * cpu_utilization * np.random.uniform(0.8, 1.2)
                current_gpu_power = gpu_watts * gpu_utilization * np.random.uniform(0.8, 1.2) if gpu_total > 0 else 0
                
                # Generate task requirements based on complexity
                if task_complexity == "Light":
                    req_cpu = np.random.randint(1, 4)
                    req_gpu = np.random.randint(0, 256) if gpu_total > 0 else 0
                    req_vram = np.random.uniform(0, 4) if gpu_total > 0 else 0
                    req_ram = np.random.uniform(1, 8)
                elif task_complexity == "Mixed":
                    req_cpu = np.random.randint(1, 8)
                    req_gpu = np.random.randint(0, 1024) if gpu_total > 0 else 0
                    req_vram = np.random.uniform(0, 12) if gpu_total > 0 else 0
                    req_ram = np.random.uniform(2, 32)
                else:  # Heavy
                    req_cpu = np.random.randint(4, 16)
                    req_gpu = np.random.randint(0, 2048) if gpu_total > 0 else 0
                    req_vram = np.random.uniform(0, 24) if gpu_total > 0 else 0
                    req_ram = np.random.uniform(8, 128)
                
                # Execution time
                execution_time = np.random.uniform(10, 300)
                
                task = {
                    "cpu_cores_total": cpu_total,
                    "cpu_watts": cpu_watts,
                    "gpu_cores_total": gpu_total,
                    "gpu_vram_total_gb": round(vram_total, 1),
                    "gpu_watts": gpu_watts,
                    "ram_total_gb": ram_total,
                    "cpu_cores_used": cpu_used,
                    "gpu_cores_used": gpu_used,
                    "gpu_vram_used_gb": round(vram_used, 1),
                    "ram_used_gb": round(ram_used, 1),
                    "current_cpu_power_w": round(current_cpu_power, 2),
                    "current_gpu_power_w": round(current_gpu_power, 2),
                    "req_cpu_cores": req_cpu,
                    "req_gpu_cores": req_gpu,
                    "req_gpu_vram_gb": round(req_vram, 1),
                    "req_ram_gb": round(req_ram, 1),
                    "execution_time": round(execution_time, 2)
                }
                sample_tasks.append(task)
            
            tasks_df = pd.DataFrame(sample_tasks)
            st.success(f"Generated {num_tasks} sample tasks with full machine state data")
            st.dataframe(tasks_df, use_container_width=True)

# Real-time Simulation
if 'tasks_df' in locals() and len(st.session_state.machines) > 0:
    st.header("🚀 Real-Time Simulation")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        if st.button("▶️ Start Real-Time Simulation", use_container_width=True, type="primary", disabled=st.session_state.simulation_running):
            st.session_state.simulation_running = True
            st.session_state.simulation_results = []
            
            progress_container = st.container()
            live_results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_task_info = st.empty()
            
            with live_results_container:
                st.subheader("Live Task Processing")
                results_table = st.empty()
                
                for i, task in tasks_df.iterrows():
                    progress_bar.progress((i + 1) / len(tasks_df))
                    status_text.text(f"Processing Task {i+1}/{len(tasks_df)}")
                    
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
                        result = {"task_id": i+1, "status": "❌ No free machines", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A"}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    capable_machines = [m for m in free_machines if m.can_accommodate(task)]
                    
                    if not capable_machines:
                        result = {"task_id": i+1, "status": "❌ No suitable free machine", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A"}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    # SLA prediction for each capable machine
                    sla_safe_machines = []
                    sla_predictions = {}
                    
                    for machine in capable_machines:
                        if model is not None:
                            features = pd.DataFrame([{
                                "cpu_cores_total": machine.specs["cpu_cores_total"],
                                "cpu_watts": machine.specs["cpu_watts"],
                                "gpu_cores_total": machine.specs["gpu_cores_total"],
                                "gpu_vram_total_gb": machine.specs["gpu_vram_total_gb"],
                                "gpu_watts": machine.specs["gpu_watts"],
                                "ram_total_gb": machine.specs["ram_total_gb"],
                                "cpu_cores_used": machine.specs["cpu_cores_total"] - machine.free_resources["cpu_cores_total"],
                                "gpu_cores_used": machine.specs["gpu_cores_total"] - machine.free_resources["gpu_cores_total"],
                                "gpu_vram_used_gb": machine.specs["gpu_vram_total_gb"] - machine.free_resources["gpu_vram_total_gb"],
                                "ram_used_gb": machine.specs["ram_total_gb"] - machine.free_resources["ram_total_gb"],
                                "current_cpu_power_w": 50.0,
                                "current_gpu_power_w": 100.0,
                                "req_cpu_cores": task.get("req_cpu_cores", 0),
                                "req_gpu_cores": task.get("req_gpu_cores", 0),
                                "req_gpu_vram_gb": task.get("req_gpu_vram_gb", 0),
                                "req_ram_gb": task.get("req_ram_gb", 0),
                            }])
                            
                            try:
                                if hasattr(model, 'predict'):
                                    prediction = model.predict(features.values)[0]
                                    sla_violation_class = int(prediction >= 0.5) if isinstance(prediction, float) else int(prediction)
                                    sla_predictions[machine.name] = prediction if isinstance(prediction, float) else float(prediction)
                                    if sla_violation_class == 0:
                                        sla_safe_machines.append(machine)
                                elif hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(features)[0]
                                    violation_probability = proba[1] if len(proba) > 1 else proba[0]
                                    sla_predictions[machine.name] = violation_probability
                                    if violation_probability < 0.5:
                                        sla_safe_machines.append(machine)
                                else:
                                    sla_predictions[machine.name] = 0.0
                                    sla_safe_machines.append(machine)
                            except Exception:
                                sla_predictions[machine.name] = 0.0
                                sla_safe_machines.append(machine)
                        else:
                            sla_predictions[machine.name] = 0.0
                            sla_safe_machines.append(machine)
                    
                    if not sla_safe_machines:
                        result = {"task_id": i+1, "status": "⚠️ All machines predict SLA violation", "machine": None, "sla_prediction": "High", "energy_score": "N/A", "fit_score": "N/A"}
                        st.session_state.simulation_results.append(result)
                        results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                        results_table.dataframe(results_df_temp, use_container_width=True)
                        st.session_state.current_time += task_interval
                        time.sleep(0.5 / simulation_speed)
                        continue
                    
                    # FIXED: Use the optimized prioritize_machines function
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
                            alpha=alpha,
                            ram_power_per_gb=0.2
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
                                "fit_score": f"{fit_score:.3f}"
                            }
                        else:
                            result = {"task_id": i+1, "status": "❌ Optimization failed", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A"}
                    except Exception as e:
                        result = {"task_id": i+1, "status": f"❌ Error: {str(e)[:50]}", "machine": None, "sla_prediction": "N/A", "energy_score": "N/A", "fit_score": "N/A"}
                    
                    st.session_state.simulation_results.append(result)
                    results_df_temp = pd.DataFrame(st.session_state.simulation_results)
                    results_table.dataframe(results_df_temp, use_container_width=True)
                    update_dashboard()
                    st.session_state.current_time += task_interval
                    time.sleep(1.0 / simulation_speed)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Real-time simulation completed!")
                current_task_info.success("All tasks processed!")
                
            st.session_state.simulation_running = False
    
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
    
    tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Machine Efficiency", "Cost Analysis"])
    
    with tab1:
        results_df = pd.DataFrame(st.session_state.simulation_results)
        successful_results = results_df[results_df['status'] == '✅ Scheduled']
        
        if not successful_results.empty:
            try:
                sla_risks = pd.to_numeric(successful_results['sla_prediction'], errors='coerce').dropna()
                energy_scores = pd.to_numeric(successful_results['energy_score'], errors='coerce').dropna()
                fit_scores = pd.to_numeric(successful_results['fit_score'], errors='coerce').dropna()
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    if len(sla_risks) > 0:
                        fig_sla_hist = px.histogram(x=sla_risks, nbins=20, title="SLA Risk Distribution", labels={'x': 'SLA Risk Score', 'y': 'Count'})
                        st.plotly_chart(fig_sla_hist, use_container_width=True, key=f"sla_histogram_{hash(str(sla_risks.values))}")
                
                with col_p2:
                    if len(energy_scores) > 0 and len(fit_scores) > 0:
                        fig_scatter = px.scatter(x=energy_scores, y=fit_scores, title="Energy vs Fit Scores", 
                                               labels={'x': 'Energy Score', 'y': 'Fit Score'})
                        st.plotly_chart(fig_scatter, use_container_width=True, key=f"energy_fit_scatter_{hash(str(energy_scores.values) + str(fit_scores.values))}")
                
                # Display metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    if len(sla_risks) > 0:
                        st.metric("Average SLA Risk", f"{sla_risks.mean():.3f}")
                with col_m2:
                    if len(energy_scores) > 0:
                        st.metric("Average Energy Score", f"{energy_scores.mean():.3f}")
                with col_m3:
                    if len(fit_scores) > 0:
                        st.metric("Average Fit Score", f"{fit_scores.mean():.3f}")
                        
            except:
                st.info("Analysis data not available yet")
        else:
            st.info("No successful tasks to analyze")
    
    with tab2:
        if st.session_state.machines:
            efficiency_data = []
            for machine in st.session_state.machines:
                if machine.utilization_history:
                    avg_cpu = np.mean([h['cpu_util'] for h in machine.utilization_history])
                    avg_gpu = np.mean([h['gpu_util'] for h in machine.utilization_history])
                    avg_ram = np.mean([h['ram_util'] for h in machine.utilization_history])
                    total_tasks = sum([h['active_tasks'] for h in machine.utilization_history])
                    
                    efficiency_data.append({
                        'Machine': machine.name,
                        'Avg CPU Util': avg_cpu * 100,
                        'Avg GPU Util': avg_gpu * 100,
                        'Avg RAM Util': avg_ram * 100,
                        'Total Task Time': total_tasks,
                        'Energy Consumed': machine.energy_consumed,
                        'Cost': machine.total_cost
                    })
            
            if efficiency_data:
                eff_df = pd.DataFrame(efficiency_data)
                
                col_e1, col_e2 = st.columns(2)
                
                with col_e1:
                    fig_util = px.bar(eff_df, x='Machine', y=['Avg CPU Util', 'Avg GPU Util', 'Avg RAM Util'], title="Average Resource Utilization", barmode='group')
                    st.plotly_chart(fig_util, use_container_width=True, key=f"utilization_bar_{hash(str(eff_df.values.tobytes()))}")
                
                with col_e2:
                    fig_efficiency = px.scatter(eff_df, x='Energy Consumed', y='Total Task Time', size='Cost', hover_name='Machine', title="Energy vs Performance")
                    st.plotly_chart(fig_efficiency, use_container_width=True, key=f"efficiency_scatter_{hash(str(eff_df.values.tobytes()))}")
                
                st.subheader("Machine Efficiency Summary")
                st.dataframe(eff_df, use_container_width=True)
            else:
                st.info("No machine efficiency data available")
        else:
            st.info("No machines to analyze")
    
    with tab3:
        if st.session_state.machines:
            cost_data = []
            for machine in st.session_state.machines:
                assigned_tasks = len([r for r in st.session_state.simulation_results if r.get('machine') == machine.name and r.get('status') == '✅ Scheduled'])
                cost_per_task = machine.total_cost / max(assigned_tasks, 1)
                
                cost_data.append({
                    'Machine': machine.name,
                    'Total Cost': machine.total_cost,
                    'Tasks Assigned': assigned_tasks,
                    'Cost per Task': cost_per_task,
                    'Energy Cost': machine.energy_consumed * 0.12,
                    'Efficiency Score': assigned_tasks / max(machine.total_cost, 0.01)
                })
            
            cost_df = pd.DataFrame(cost_data)
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                fig_cost = px.bar(cost_df, x='Machine', y='Total Cost', title="Total Cost by Machine")
                st.plotly_chart(fig_cost, use_container_width=True, key=f"cost_bar_{hash(str(cost_df.values.tobytes()))}")
            
            with col_c2:
                fig_efficiency = px.scatter(cost_df, x='Cost per Task', y='Efficiency Score', size='Tasks Assigned', hover_name='Machine', title="Cost Efficiency Analysis")
                st.plotly_chart(fig_efficiency, use_container_width=True, key=f"cost_efficiency_{hash(str(cost_df.values.tobytes()))}")
            
            st.subheader("Cost Breakdown")
            st.dataframe(cost_df, use_container_width=True)
            
            st.subheader("Cost Breakdown")
            st.dataframe(cost_df, use_container_width=True)
        else:
            st.info("No cost data available")

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
### 🚀 Real-Time Features:
- **Live Dashboard**: Machine utilization updates in real-time during simulation
- **Instant Feedback**: Task assignments show immediately as they're processed  
- **Dynamic Visualizations**: Charts update continuously showing current state
- **Real-Time Metrics**: Success rates, costs, and energy consumption update live
- **Progressive Results**: See results as they happen, not just at the end

### 🔧 Technical Implementation:
- **Binary SLA Classification**: LightGBM model predicts 0 (safe) or 1 (violation)
- **Multi-Objective Optimization**: Balances energy efficiency with resource fit using imported prioritize_machines function
- **Resource Tracking**: Real-time monitoring of CPU, GPU, RAM, and task states
- **Timeline Analysis**: Historical utilization data with time-series visualization
- **Stable Chart Keys**: Fixed graph jittering with consistent plot identifiers

### ⚡ Performance Features:
- Configurable simulation speed (0.5x to 5x)
- Adjustable task intervals (1-20 seconds)
- Live progress tracking with task details
- Real-time dashboard updates during processing
- Optimized energy and fit score calculations
""")