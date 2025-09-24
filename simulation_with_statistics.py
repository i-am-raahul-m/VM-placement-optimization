import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ------------------------
# Load trained LightGBM model
# ------------------------
@st.cache_resource
def load_model():
    try:
        # Try to load as pickle file first
        return joblib.load("lightgbm_sla_model.pkl")
    except FileNotFoundError:
        try:
            # Try to load as text file (LightGBM text format)
            import lightgbm as lgb
            return lgb.Booster(model_file="lightgbm_sla_model.txt")
        except (FileNotFoundError, ImportError) as e:
            st.warning("⚠️ lightgbm_sla_model.txt not found or LightGBM not installed. Using mock predictions.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {str(e)}. Using mock predictions.")
        return None

# ------------------------
# Placement Optimization Function
# ------------------------
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
    """
    df = safe_machines_df.copy().reset_index(drop=False)
    
    # Required columns check
    required_cols = [
        "cpu_cores_total", "cpu_watts", "gpu_cores_total", "gpu_vram_total_gb", "gpu_watts",
        "ram_total_gb", "cpu_cores_used", "gpu_cores_used", "gpu_vram_used_gb", "ram_used_gb",
        "current_cpu_power_w", "current_gpu_power_w"
    ]
    
    # Add missing columns with defaults if needed
    for col in required_cols:
        if col not in df.columns:
            if "power" in col:
                df[col] = 0.0
            elif "used" in col:
                df[col] = 0.0
            else:
                df[col] = df.get(col.replace("_total", ""), 0.0)

    # Requested resources
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

    # Fit score calculation
    df["fit_cpu"] = np.abs(df["remaining_cpu_cores"] - r_cpu) / (df["cpu_cores_total"].replace(0, eps))
    df["fit_gpu"] = np.abs(df["remaining_gpu_cores"] - r_gpu) / (df["gpu_cores_total"].replace(0, eps))
    df["fit_vram"] = np.abs(df["remaining_gpu_vram_gb"] - r_vram) / (df["gpu_vram_total_gb"].replace(0, eps))
    df["fit_ram"] = np.abs(df["remaining_ram_gb"] - r_ram) / (df["ram_total_gb"].replace(0, eps))

    df["fit_score_raw"] = (df[["fit_cpu", "fit_gpu", "fit_vram", "fit_ram"]].sum(axis=1)) / 4.0

    # Normalize fit score
    min_fit = df["fit_score_raw"].min()
    max_fit = df["fit_score_raw"].max()
    if abs(max_fit - min_fit) < eps:
        df["fit_score"] = 0.0
    else:
        df["fit_score"] = (df["fit_score_raw"] - min_fit) / (max_fit - min_fit)

    # Energy estimation
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
            ram_per_gb = max(0.05, min(2.0, (cur_power * 0.02) / total_ram_gb))
            
        inc_power = cpu_per_core * r_cpu + gpu_per_core * r_gpu + gpu_vram_per_gb * r_vram + ram_per_gb * r_ram
        return inc_power

    df["est_incremental_power_w"] = df.apply(derive_row_power, axis=1)

    # Normalize energy score
    min_e = df["est_incremental_power_w"].min()
    max_e = df["est_incremental_power_w"].max()
    if abs(max_e - min_e) < eps:
        df["energy_score"] = 0.0
    else:
        df["energy_score"] = (df["est_incremental_power_w"] - min_e) / (max_e - min_e)

    # Combine scores
    alpha = float(alpha)
    df["final_score"] = alpha * df["energy_score"] + (1.0 - alpha) * df["fit_score"]

    # Sort by best score
    ranked = df.sort_values(["final_score", "est_incremental_power_w", "fit_score"]).reset_index(drop=True)
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
# Enhanced Machine class
# ------------------------
class Machine:
    def __init__(self, specs, name):
        self.name = name
        self.specs = specs.copy()
        self.free_resources = specs.copy()
        self.tasks = []  # list of (task_id, task, start_time, finish_time)
        self.utilization_history = []
        self.energy_consumed = 0
        self.total_cost = 0
        self.sla_violations = 0

    def update_status(self, current_time):
        """Update machine status and free up finished tasks"""
        finished_tasks = []
        for task_data in self.tasks[:]:
            task_id, task, start_time, finish_time = task_data
            if finish_time <= current_time:
                self.release_task(task)
                finished_tasks.append(task_data)
                self.tasks.remove(task_data)
        
        # Record utilization
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
        """Check if machine can accommodate the task"""
        return (self.free_resources["cpu_cores_total"] >= task.get("req_cpu_cores", 0) and
                self.free_resources["gpu_cores_total"] >= task.get("req_gpu_cores", 0) and
                self.free_resources["gpu_vram_total_gb"] >= task.get("req_gpu_vram_gb", 0) and
                self.free_resources["ram_total_gb"] >= task.get("req_ram_gb", 0))

    def is_free(self, current_time):
        """Check if machine is free and update status"""
        self.update_status(current_time)
        return len(self.tasks) == 0

    def allocate_task(self, task, current_time, task_id):
        """Allocate a task to this machine"""
        finish_time = current_time + task.get("execution_time", 1)
        self.tasks.append((task_id, task, current_time, finish_time))
        
        # Reduce available resources
        self.free_resources["cpu_cores_total"] -= task.get("req_cpu_cores", 0)
        self.free_resources["gpu_cores_total"] -= task.get("req_gpu_cores", 0)
        self.free_resources["gpu_vram_total_gb"] -= task.get("req_gpu_vram_gb", 0)
        self.free_resources["ram_total_gb"] -= task.get("req_ram_gb", 0)
        
        # Update costs and energy
        duration_hours = task.get("execution_time", 1) / 3600  # assuming time is in seconds
        self.total_cost += self.specs.get("cost_per_hour", 0) * duration_hours
        
        # Estimate energy consumption
        cpu_power = (task.get("req_cpu_cores", 0) / self.specs["cpu_cores_total"]) * self.specs["cpu_watts"]
        gpu_power = (task.get("req_gpu_cores", 0) / max(1, self.specs["gpu_cores_total"])) * self.specs["gpu_watts"]
        self.energy_consumed += (cpu_power + gpu_power) * duration_hours

    def release_task(self, task):
        """Release resources when task finishes"""
        self.free_resources["cpu_cores_total"] += task.get("req_cpu_cores", 0)
        self.free_resources["gpu_cores_total"] += task.get("req_gpu_cores", 0)
        self.free_resources["gpu_vram_total_gb"] += task.get("req_gpu_vram_gb", 0)
        self.free_resources["ram_total_gb"] += task.get("req_ram_gb", 0)

    def to_dataframe_row(self):
        """Convert machine state to DataFrame row for optimization"""
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
            "current_cpu_power_w": 0.0,  # Can be estimated based on utilization
            "current_gpu_power_w": 0.0,  # Can be estimated based on utilization
            "cost_per_hour": self.specs.get("cost_per_hour", 0),
            "energy_consumed": self.energy_consumed,
            "total_cost": self.total_cost,
            "sla_violations": self.sla_violations
        }

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="VM Placement Simulator", layout="wide")
st.title("🖥️ Enhanced VM Placement Simulator")
st.markdown("*Machine Learning-driven SLA prediction with multi-objective optimization*")

# Load model
model = load_model()

# Initialize session state
if "machines" not in st.session_state:
    st.session_state.machines = []
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = []
if "current_time" not in st.session_state:
    st.session_state.current_time = 0

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    alpha = st.slider("Energy vs Fit Weight (α)", 0.0, 1.0, 0.5, 0.1,
                      help="0 = prioritize fit, 1 = prioritize energy efficiency")
    simulation_speed = st.selectbox("Simulation Speed", [0.5, 1, 2, 5], index=1)
    task_interval = st.slider("Time Between Tasks (seconds)", 1, 20, 5, 1,
                              help="Time interval between scheduling each task")
    
    st.info("📋 **Workflow:**\n1. Check if machine is free\n2. Check if machine can accommodate task\n3. Use LightGBM for SLA prediction (0=safe, 1=violation)\n4. Apply optimization to select best machine from safe machines")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔧 Machine Pool Management")
    
    # Add preset machines
    st.subheader("Add Preset Machine")
    preset = st.selectbox("Choose preset", list(PRESETS.keys()))
    
    col_preset1, col_preset2 = st.columns(2)
    with col_preset1:
        if st.button("➕ Add Preset Machine", use_container_width=True):
            idx = len(st.session_state.machines) + 1
            st.session_state.machines.append(Machine(PRESETS[preset], f"{preset}-{idx}"))
            st.success(f"Added {preset}-{idx}")
    
    with col_preset2:
        if st.button("🗑️ Clear All Machines", use_container_width=True):
            st.session_state.machines = []
            st.session_state.simulation_results = []
            st.success("Cleared all machines")

    # Custom machine form
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
            spec = {
                "cpu_cores_total": cpu, "gpu_cores_total": gpu, "gpu_vram_total_gb": vram,
                "ram_total_gb": ram, "cpu_watts": cpu_w, "gpu_watts": gpu_w, "cost_per_hour": cost
            }
            idx = len(st.session_state.machines) + 1
            st.session_state.machines.append(Machine(spec, f"Custom-{idx}"))
            st.success(f"Added Custom-{idx}")

with col2:
    st.header("📊 Current Machine Pool")
    
    if st.session_state.machines:
        # Create summary dataframe
        machine_data = []
        for m in st.session_state.machines:
            data = m.to_dataframe_row()
            data["utilization_cpu"] = (data["cpu_cores_used"] / data["cpu_cores_total"]) * 100
            data["utilization_ram"] = (data["ram_used_gb"] / data["ram_total_gb"]) * 100
            data["active_tasks"] = len(m.tasks)
            machine_data.append(data)
        
        machines_df = pd.DataFrame(machine_data)
        
        # Display summary metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Machines", len(st.session_state.machines))
        with col_m2:
            busy_machines = sum(1 for m in st.session_state.machines if len(m.tasks) > 0)
            st.metric("Busy Machines", busy_machines)
        with col_m3:
            total_cost = sum(m.total_cost for m in st.session_state.machines)
            st.metric("Total Cost", f"${total_cost:.2f}")
        
        # Machine status table
        display_cols = ["machine_name", "cpu_cores_total", "ram_total_gb", "utilization_cpu", 
                       "utilization_ram", "active_tasks", "total_cost"]
        st.dataframe(machines_df[display_cols], use_container_width=True)
        
    else:
        st.info("No machines added yet. Add some machines to get started!")

# Workload section
st.header("📋 Workload Management")

col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.subheader("Upload Workload CSV")
    workload_file = st.file_uploader("Choose CSV file", type="csv",
                                     help="CSV should contain: req_cpu_cores, req_gpu_cores, req_gpu_vram_gb, req_ram_gb, execution_time")
    
    if workload_file:
        tasks_df = pd.read_csv(workload_file)
        st.dataframe(tasks_df.head(), use_container_width=True)
        
        st.subheader("Sample Tasks")
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
            # Generate sample tasks based on complexity
            np.random.seed(42)
            
            if task_complexity == "Light":
                cpu_range, ram_range, gpu_range = (1, 4), (1, 8), (0, 0)
            elif task_complexity == "Mixed":
                cpu_range, ram_range, gpu_range = (1, 8), (2, 16), (0, 512)
            else:  # Heavy
                cpu_range, ram_range, gpu_range = (4, 16), (8, 64), (0, 2048)
            
            sample_tasks = []
            for i in range(num_tasks):
                task = {
                    "task_id": f"task_{i+1}",
                    "req_cpu_cores": np.random.randint(cpu_range[0], cpu_range[1]+1),
                    "req_gpu_cores": np.random.randint(gpu_range[0], gpu_range[1]+1),
                    "req_gpu_vram_gb": np.random.randint(0, 16) if gpu_range[1] > 0 else 0,
                    "req_ram_gb": np.random.randint(ram_range[0], ram_range[1]+1),
                    "execution_time": np.random.randint(10, 300)  # seconds
                }
                sample_tasks.append(task)
            
            tasks_df = pd.DataFrame(sample_tasks)
            st.success(f"Generated {num_tasks} sample tasks")
            st.dataframe(tasks_df, use_container_width=True)

# Simulation section
if 'tasks_df' in locals() and len(st.session_state.machines) > 0:
    st.header("🚀 Run Simulation")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        if st.button("▶️ Start Simulation", use_container_width=True, type="primary"):
            st.session_state.simulation_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            with results_container:
                st.subheader("Simulation Progress")
                
                for i, task in tasks_df.iterrows():
                    progress_bar.progress((i + 1) / len(tasks_df))
                    status_text.text(f"Processing Task {i+1}/{len(tasks_df)}")
                    
                    current_time = st.session_state.current_time
                    
                    # Update all machines
                    for machine in st.session_state.machines:
                        machine.update_status(current_time)
                    
                    # Step 1: Find machines that are FREE (no active tasks)
                    free_machines = [m for m in st.session_state.machines 
                                   if m.is_free(current_time)]
                    
                    if not free_machines:
                        result = {
                            "task_id": i+1,
                            "status": "❌ No free machines",
                            "machine": None,
                            "sla_prediction": "N/A",
                            "energy_score": "N/A",
                            "fit_score": "N/A"
                        }
                        st.session_state.simulation_results.append(result)
                        continue
                    
                    # Step 2: Among free machines, check which can accommodate the task
                    capable_machines = [m for m in free_machines 
                                      if m.can_accommodate(task)]
                    
                    if not capable_machines:
                        result = {
                            "task_id": i+1,
                            "status": "❌ No suitable free machine",
                            "machine": None,
                            "sla_prediction": "N/A",
                            "energy_score": "N/A",
                            "fit_score": "N/A"
                        }
                        st.session_state.simulation_results.append(result)
                        continue
                    
                    # Step 3: Use LightGBM for binary classification (0=safe, 1=violation)
                    sla_safe_machines = []
                    sla_predictions = {}
                    
                    for machine in capable_machines:
                        if model is not None:
                            # Create feature vector for prediction
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
                                "current_cpu_power_w": 50.0,  # Mock value
                                "current_gpu_power_w": 100.0,  # Mock value
                                "req_cpu_cores": task.get("req_cpu_cores", 0),
                                "req_gpu_cores": task.get("req_gpu_cores", 0),
                                "req_gpu_vram_gb": task.get("req_gpu_vram_gb", 0),
                                "req_ram_gb": task.get("req_ram_gb", 0),
                            }])
                            
                            try:
                                # Handle different model types for binary classification
                                if hasattr(model, 'predict'):
                                    # Binary classification: 0 = no violation, 1 = violation
                                    prediction = model.predict(features.values)[0] if hasattr(model, 'predict') else 0
                                    sla_violation_class = int(prediction >= 0.5) if isinstance(prediction, float) else int(prediction)
                                    
                                    # Store the raw prediction for display
                                    sla_predictions[machine.name] = prediction if isinstance(prediction, float) else float(prediction)
                                    
                                    # Only include machines where model predicts NO SLA violation (class 0)
                                    if sla_violation_class == 0:
                                        sla_safe_machines.append(machine)
                                elif hasattr(model, 'predict_proba'):
                                    # If using sklearn-style model with predict_proba
                                    proba = model.predict_proba(features)[0]
                                    violation_probability = proba[1] if len(proba) > 1 else proba[0]
                                    sla_predictions[machine.name] = violation_probability
                                    
                                    # Use probability < 0.5 as "safe" (no violation predicted)
                                    if violation_probability < 0.5:
                                        sla_safe_machines.append(machine)
                                else:
                                    # Fallback: assume safe (no SLA violation)
                                    sla_predictions[machine.name] = 0.0
                                    sla_safe_machines.append(machine)
                                    
                            except Exception as e:
                                # Fallback: assume safe (no SLA violation)
                                sla_predictions[machine.name] = 0.0
                                sla_safe_machines.append(machine)
                        else:
                            # No model available, assume all capable machines are safe
                            sla_predictions[machine.name] = 0.0
                            sla_safe_machines.append(machine)
                    
                    if not sla_safe_machines:
                        result = {
                            "task_id": i+1,
                            "status": "⚠️ All machines predict SLA violation",
                            "machine": None,
                            "sla_prediction": "High",
                            "energy_score": "N/A",
                            "fit_score": "N/A"
                        }
                        st.session_state.simulation_results.append(result)
                        continue
                    
                    # Step 4: Apply optimization to select the best machine from SLA-safe machines
                    safe_machines_df = pd.DataFrame([m.to_dataframe_row() for m in sla_safe_machines])
                    task_reqs = {
                        "req_cpu_cores": task.get("req_cpu_cores", 0),
                        "req_gpu_cores": task.get("req_gpu_cores", 0),
                        "req_gpu_vram_gb": task.get("req_gpu_vram_gb", 0),
                        "req_ram_gb": task.get("req_ram_gb", 0)
                    }
                    
                    try:
                        ranked_machines, best_machine_row = prioritize_machines(
                            safe_machines_df, task_reqs, alpha=alpha
                        )
                        
                        if best_machine_row is not None:
                            # Find the actual machine object
                            chosen_machine = next(m for m in sla_safe_machines 
                                                if m.name == best_machine_row["machine_name"])
                            
                            # Allocate task
                            chosen_machine.allocate_task(task, current_time, f"task_{i+1}")
                            
                            result = {
                                "task_id": i+1,
                                "status": "✅ Scheduled",
                                "machine": chosen_machine.name,
                                "sla_prediction": f"{sla_predictions[chosen_machine.name]:.3f}",
                                "energy_score": f"{best_machine_row.get('energy_score', 0):.3f}",
                                "fit_score": f"{best_machine_row.get('fit_score', 0):.3f}"
                            }
                        else:
                            result = {
                                "task_id": i+1,
                                "status": "❌ Optimization failed",
                                "machine": None,
                                "sla_prediction": "N/A",
                                "energy_score": "N/A",
                                "fit_score": "N/A"
                            }
                    except Exception as e:
                        result = {
                            "task_id": i+1,
                            "status": f"❌ Error: {str(e)[:50]}",
                            "machine": None,
                            "sla_prediction": "N/A",
                            "energy_score": "N/A",
                            "fit_score": "N/A"
                        }
                    
                    st.session_state.simulation_results.append(result)
                    
                    # Update simulation time using the configurable task interval
                    st.session_state.current_time += task_interval
                    
                    # Small delay for visualization (based on simulation speed)
                    time.sleep(0.1 / simulation_speed)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Simulation completed!")
    
    with col_sim2:
        if st.button("⏹️ Reset Simulation", use_container_width=True):
            st.session_state.simulation_results = []
            st.session_state.current_time = 0
            # Reset all machines
            for machine in st.session_state.machines:
                machine.tasks = []
                machine.free_resources = machine.specs.copy()
                machine.utilization_history = []
                machine.energy_consumed = 0
                machine.total_cost = 0
                machine.sla_violations = 0
            st.success("Simulation reset successfully!")

# Results section
if st.session_state.simulation_results:
    st.header("📈 Simulation Results")
    
    results_df = pd.DataFrame(st.session_state.simulation_results)
    
    # Summary metrics
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        successful_tasks = len(results_df[results_df['status'] == '✅ Scheduled'])
        st.metric("Successful Tasks", successful_tasks)
    
    with col_r2:
        failed_tasks = len(results_df) - successful_tasks
        st.metric("Failed Tasks", failed_tasks)
    
    with col_r3:
        if successful_tasks > 0:
            success_rate = (successful_tasks / len(results_df)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "0%")
    
    with col_r4:
        total_energy = sum(m.energy_consumed for m in st.session_state.machines)
        st.metric("Total Energy", f"{total_energy:.2f} kWh")
    
    # Detailed results table
    st.subheader("Task Assignment Details")
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Task Distribution", "Machine Utilization", "Performance Metrics"])
    
    with tab1:
        if successful_tasks > 0:
            # Task distribution by machine
            successful_results = results_df[results_df['status'] == '✅ Scheduled']
            machine_counts = successful_results['machine'].value_counts()
            
            fig_pie = px.pie(
                values=machine_counts.values,
                names=machine_counts.index,
                title="Task Distribution Across Machines"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Timeline of task assignments
            successful_results_with_time = successful_results.copy()
            successful_results_with_time['time'] = range(len(successful_results_with_time))
            
            fig_timeline = px.scatter(
                successful_results_with_time,
                x='time',
                y='machine',
                color='sla_prediction',
                size=[10] * len(successful_results_with_time),
                title="Task Assignment Timeline",
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No successful task assignments to visualize.")
    
    with tab2:
        if st.session_state.machines:
            # Machine utilization overview
            machine_util_data = []
            for machine in st.session_state.machines:
                if machine.utilization_history:
                    latest_util = machine.utilization_history[-1]
                    machine_util_data.append({
                        'Machine': machine.name,
                        'CPU Utilization': latest_util['cpu_util'] * 100,
                        'GPU Utilization': latest_util['gpu_util'] * 100,
                        'RAM Utilization': latest_util['ram_util'] * 100,
                        'Active Tasks': latest_util['active_tasks']
                    })
            
            if machine_util_data:
                util_df = pd.DataFrame(machine_util_data)
                
                # Utilization heatmap
                fig_heatmap = px.bar(
                    util_df,
                    x='Machine',
                    y=['CPU Utilization', 'GPU Utilization', 'RAM Utilization'],
                    title="Current Resource Utilization by Machine",
                    barmode='group'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Active tasks distribution
                fig_tasks = px.bar(
                    util_df,
                    x='Machine',
                    y='Active Tasks',
                    title="Active Tasks per Machine"
                )
                st.plotly_chart(fig_tasks, use_container_width=True)
            else:
                st.info("No utilization data available yet.")
        else:
            st.info("No machines to display utilization for.")
    
    with tab3:
        if successful_tasks > 0:
            # Performance metrics
            successful_results = results_df[results_df['status'] == '✅ Scheduled']
            
            # Convert string metrics to float for analysis
            try:
                sla_predictions_numeric = pd.to_numeric(successful_results['sla_prediction'], errors='coerce').dropna()
                energy_scores = pd.to_numeric(successful_results['energy_score'], errors='coerce').dropna()
                fit_scores = pd.to_numeric(successful_results['fit_score'], errors='coerce').dropna()
                
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    # SLA prediction distribution
                    if len(sla_predictions_numeric) > 0:
                        fig_sla = px.histogram(
                            x=sla_predictions_numeric,
                            nbins=20,
                            title="SLA Prediction Distribution",
                            labels={'x': 'SLA Prediction Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_sla, use_container_width=True)
                
                with col_m2:
                    # Energy vs Fit score scatter
                    if len(energy_scores) > 0 and len(fit_scores) > 0:
                        fig_scatter = px.scatter(
                            x=energy_scores,
                            y=fit_scores,
                            title="Energy Efficiency vs Resource Fit",
                            labels={'x': 'Energy Score', 'y': 'Fit Score'}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Summary statistics
                st.subheader("Performance Summary")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    if len(sla_predictions_numeric) > 0:
                        avg_sla_prediction = sla_predictions_numeric.mean()
                        st.metric("Avg SLA Prediction", f"{avg_sla_prediction:.3f}")
                        high_risk_count = len(sla_predictions_numeric[sla_predictions_numeric > 0.5])
                        st.metric("High Risk Tasks", high_risk_count)
                
                with col_s2:
                    total_cost = sum(m.total_cost for m in st.session_state.machines)
                    st.metric("Total Cost", f"${total_cost:.2f}")
                    if len(energy_scores) > 0:
                        avg_energy = energy_scores.mean()
                        st.metric("Avg Energy Score", f"{avg_energy:.3f}")
                
                with col_s3:
                    if len(fit_scores) > 0:
                        avg_fit = fit_scores.mean()
                        st.metric("Avg Fit Score", f"{avg_fit:.3f}")
                    total_energy = sum(m.energy_consumed for m in st.session_state.machines)
                    st.metric("Total Energy", f"{total_energy:.2f} kWh")
            
            except Exception as e:
                st.error(f"Error analyzing performance metrics: {str(e)}")
        else:
            st.info("No successful tasks to analyze.")

# Export results
if st.session_state.simulation_results:
    st.header("💾 Export Results")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        if st.button("📊 Download Results CSV"):
            results_df = pd.DataFrame(st.session_state.simulation_results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"vm_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_e2:
        if st.button("🖥️ Download Machine Stats CSV"):
            machine_data = [m.to_dataframe_row() for m in st.session_state.machines]
            machines_df = pd.DataFrame(machine_data)
            csv = machines_df.to_csv(index=False)
            st.download_button(
                label="Download Machine Stats CSV",
                data=csv,
                file_name=f"machine_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
### 🚀 Features:
- **ML-Powered SLA Prediction**: Uses LightGBM binary classification (0=safe, 1=violation)
- **Multi-Objective Optimization**: Balances energy efficiency and resource fit
- **Real-time Visualization**: Interactive charts and metrics
- **Comprehensive Analytics**: Detailed performance analysis and reporting
- **Flexible Configuration**: Customizable machine types, optimization parameters, and timing controls

### 📋 CSV Format:
Your workload CSV should contain columns: `req_cpu_cores`, `req_gpu_cores`, `req_gpu_vram_gb`, `req_ram_gb`, `execution_time`

### ⚙️ Configuration Options:
- **Simulation Speed**: Controls visualization speed (0.5x to 5x)
- **Task Interval**: Time between scheduling tasks (1-20 seconds)
- **Energy vs Fit Weight**: Balance between energy efficiency and resource fit

### 🤖 Model Information:
- **Binary Classification**: 0 = No SLA violation predicted, 1 = SLA violation predicted
- **Safe Machines**: Only machines with prediction = 0 are considered for optimization
- **Fallback**: If model unavailable, all capable machines are considered safe
""")