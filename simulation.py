import streamlit as st
import pandas as pd
import time
import numpy as np

# ------------------------
# Preset machine templates
# ------------------------
PRESETS = {
    "Small VM": {"cpu_cores_total": 4, "gpu_cores_total": 0, "gpu_vram_total_gb": 0, "ram_total_gb": 8,
                 "cpu_watts": 65, "gpu_watts": 0},
    "GPU Node": {"cpu_cores_total": 8, "gpu_cores_total": 2048, "gpu_vram_total_gb": 16, "ram_total_gb": 32,
                 "cpu_watts": 95, "gpu_watts": 250},
    "Balanced": {"cpu_cores_total": 16, "gpu_cores_total": 1024, "gpu_vram_total_gb": 8, "ram_total_gb": 64,
                 "cpu_watts": 120, "gpu_watts": 150},
    "High RAM": {"cpu_cores_total": 8, "gpu_cores_total": 0, "gpu_vram_total_gb": 0, "ram_total_gb": 128,
                 "cpu_watts": 85, "gpu_watts": 0},
    "Beast": {"cpu_cores_total": 32, "gpu_cores_total": 4096, "gpu_vram_total_gb": 48, "ram_total_gb": 256,
              "cpu_watts": 200, "gpu_watts": 400},
}

# ------------------------
# Helper classes
# ------------------------
class Machine:
    def __init__(self, specs, name):
        self.name = name
        self.specs = specs.copy()
        self.free_resources = specs.copy()
        self.tasks = []  # list of (task, finish_time)

    def is_free(self, current_time):
        # Free up finished tasks
        self.tasks = [(t, ft) for (t, ft) in self.tasks if ft > current_time]
        # If no task is running, it’s free
        return len(self.tasks) == 0

    def allocate(self, task, current_time):
        finish_time = current_time + task["execution_time"]
        self.tasks.append((task, finish_time))
        # reduce resources
        self.free_resources["cpu_cores_total"] -= task["req_cpu_cores"]
        self.free_resources["gpu_cores_total"] -= task["req_gpu_cores"]
        self.free_resources["gpu_vram_total_gb"] -= task["req_gpu_vram_gb"]
        self.free_resources["ram_total_gb"] -= task["req_ram_gb"]

    def release(self, task):
        # restore resources
        self.free_resources["cpu_cores_total"] += task["req_cpu_cores"]
        self.free_resources["gpu_cores_total"] += task["req_gpu_cores"]
        self.free_resources["gpu_vram_total_gb"] += task["req_gpu_vram_gb"]
        self.free_resources["ram_total_gb"] += task["req_ram_gb"]

# ------------------------
# Multi-objective optimizer
# ------------------------
def score_machine(machine, task):
    # Energy = CPU watts * fraction + GPU watts * fraction
    frac_cpu = task["req_cpu_cores"] / (machine.specs["cpu_cores_total"] + 1e-6)
    frac_gpu = task["req_gpu_cores"] / (machine.specs["gpu_cores_total"] + 1e-6) if machine.specs["gpu_cores_total"] else 0
    frac_ram = task["req_ram_gb"] / (machine.specs["ram_total_gb"] + 1e-6)

    energy = frac_cpu * machine.specs["cpu_watts"] + frac_gpu * machine.specs["gpu_watts"]

    # Fit penalty
    diffs = []
    reqs = {"cpu_cores_total": "req_cpu_cores", "gpu_cores_total": "req_gpu_cores", "gpu_vram_total_gb": "req_gpu_vram_gb", "ram_total_gb": "req_ram_gb"}
    for res in ["cpu_cores_total", "gpu_cores_total", "gpu_vram_total_gb", "ram_total_gb"]:
        avail = machine.free_resources[res] + 1e-6
        req = task[reqs[res]]
        if req == 0: 
            continue
        diffs.append(((req/avail) - 1) ** 2)
    fit_penalty = np.mean(diffs) if diffs else 0

    return energy, fit_penalty

def pick_best_machine(machines, task, alpha=0.5):
    scored = []
    energies, fits = [], []
    for m in machines:
        e, f = score_machine(m, task)
        scored.append((m, e, f))
        energies.append(e)
        fits.append(f)

    if not scored:
        return None

    # normalize
    e_min, e_max = min(energies), max(energies)
    f_min, f_max = min(fits), max(fits)
    results = []
    for m, e, f in scored:
        e_norm = (e - e_min) / (e_max - e_min + 1e-6)
        f_norm = (f - f_min) / (f_max - f_min + 1e-6)
        score = alpha * (1 - e_norm) + (1 - alpha) * (1 - f_norm)
        results.append((m, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0]

# ------------------------
# Streamlit app
# ------------------------
st.title("VM Placement Simulator")

# Machine pool
if "machines" not in st.session_state:
    st.session_state.machines = []

st.header("Add Machines")
preset = st.selectbox("Choose preset", list(PRESETS.keys()))
if st.button("Add Preset Machine"):
    idx = len(st.session_state.machines) + 1
    st.session_state.machines.append(Machine(PRESETS[preset], f"{preset}-{idx}"))

st.subheader("Or Add Custom Machine")
with st.form("custom_form"):
    cpu = st.number_input("CPU cores", 1, 128, 8)
    gpu = st.number_input("GPU cores", 0, 8192, 0)
    vram = st.number_input("GPU VRAM (GB)", 0, 128, 0)
    ram = st.number_input("RAM (GB)", 1, 512, 16)
    cpu_w = st.number_input("CPU Watts", 1, 400, 95)
    gpu_w = st.number_input("GPU Watts", 0, 600, 0)
    submitted = st.form_submit_button("Add Custom Machine")
    if submitted:
        spec = {"cpu_cores_total": cpu, "gpu_cores_total": gpu, "gpu_vram_total_gb": vram,
                "ram_total_gb": ram, "cpu_watts": cpu_w, "gpu_watts": gpu_w}
        idx = len(st.session_state.machines) + 1
        st.session_state.machines.append(Machine(spec, f"Custom-{idx}"))

st.write("### Current Pool")
for m in st.session_state.machines:
    st.json({"name": m.name, **m.specs})

# Workload
st.header("Load Workload CSV")
workload_file = st.file_uploader("Upload CSV", type="csv")

if workload_file:
    tasks = pd.read_csv(workload_file)
    st.dataframe(tasks)

    if st.button("Start Simulation"):
        current_time = 0
        for i, task in tasks.iterrows():
            st.write(f"### Scheduling Task {i+1}")
            free_machines = [m for m in st.session_state.machines if m.is_free(current_time)]

            if not free_machines:
                st.error("No free machine available right now. Task queued.")
            else:
                # Step 1: (binary classifier stub — for now assume all free machines are SLA safe)
                safe_machines = free_machines

                # Step 2: optimizer
                chosen = pick_best_machine(safe_machines, task, alpha=0.5)

                if chosen:
                    chosen.allocate(task, current_time)
                    st.success(f"Task {i+1} placed on {chosen.name}")
                else:
                    st.error("No suitable machine found for task.")

            # simulate wait
            time.sleep(5)
            current_time += 5
