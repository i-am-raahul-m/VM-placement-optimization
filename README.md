# 🖥️ Real-Time Workload Simulation Dashboard

A **Streamlit-powered dashboard** to simulate workloads, predict SLA violations, and optimize machine usage using a variety of **Machine Learning models**.

---
## 👥 Team Members

- **Tarun Srikumar** — 23BCE1032  
- **Derrick Richard** — 23BCE1040  
- **Ram Sundar Radhakrishnan** — 23BCE1939  
- **Raahul M** — 23BCE5106

---
## 🚀 Overview

This dashboard allows users to:

- Simulate machine workloads in real time  
- Predict **SLA violations** with multiple ML models  
- Manage machine pools (preset & custom configurations)  
- Upload or generate workloads (CSV or synthetic)  
- Optimize machine allocation with **multi-objective optimization**  
- Analyze model & system performance with live dashboards  
- Export results for further analysis  

---

## 📊 Features

### 🔧 Machine Management
- Add **Preset** or **Custom Machines** with CPU, GPU, RAM, wattage, and cost  
- Clear all machines  
- View current pool and utilization  

### 📂 Workload Management
- Upload workload from **CSV**  
- Generate synthetic workloads for testing  
- Metrics tracked:
  - Total tasks
  - Average CPU/RAM usage
  - SLA predictions
  - Execution times  

### ⚡ Real-Time Simulation
- Reset previous runs  
- Loop through tasks & update machine states  
- Predict SLA violations with trained ML models  
- Perform **multi-objective optimization** for machine allocation  

### 📈 Live Results
- Metrics displayed:
  - Tasks processed
  - SLA success rate
  - Cost & energy consumption
- Task logs (last 10 tasks with status)  

### 📉 Advanced Analytics
- **Performance Analysis**: SLA distribution, energy vs fit, high-risk tasks  
- **Machine Efficiency**: CPU/GPU/RAM utilization, active tasks per machine  
- **ML Model Comparison**:
  - SLA prediction boxplots
  - Task success metrics  

### 📤 Export
- Export simulation results  
- Export machine data & utilization history  

---

## 🤖 Machine Learning Models

The dashboard integrates and compares multiple models:

| Model                          | Accuracy |
|--------------------------------|----------|
| MLP (Neural Net)               | **0.9872** |
| TabNet                         | 0.9538 |
| XGBoost                        | 0.9372 |
| LightGBM                       | 0.9343 |
| CatBoost                       | 0.921 |
| GradientBoostingClassifier     | 0.923 |
| Random Forest                  | 0.8708 |
| Ridge Classifier               | 0.7748 |
| TabTransformer                 | 0.792 |
| AdaBoost                       | 0.7628 |
| SGD Classifier                 | 0.6486 |

---

## 🛠️ Tech Stack

- **Frontend & Dashboard**: [Streamlit](https://streamlit.io/)  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Plotly (Express, Graph Objects, Subplots)  
- **ML Models**:  
  - Scikit-learn: Ridge, SGD, Random Forest, GradientBoost, AdaBoost  
  - LightGBM, XGBoost, CatBoost  
  - PyTorch: MLP, TabTransformer, TabNet  

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/i-am-raahul-m/VM-placement-optimization.git
cd VM-placement-optimization

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
