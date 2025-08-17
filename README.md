# NASA-Hydraulics
# EcoHydra — NASA Open-Data Energy & CO₂ Advisor for Hydraulic Systems

> Turns NASA climate data into **actionable savings** by comparing **baseline vs. upgraded** hydraulic configurations — with **$ saved/yr, Payback, NPV, IRR**.

[🎥 Demo Video](<ADD_LINK>) • [🌐 Live App](<ADD_LINK>) • [📦 GitHub Repo](<THIS_REPO_URL>) • [📊 Exec Summary Excel](./docs/ExecutiveSummary.xlsx)

---

## 1) Problem
Hydraulic machines shed a lot of energy as heat. Approving upgrades (servo/proportional control, finer filtration, biodegradable oil) is hard without **local climate context** and a **finance-grade** business case.

## 2) Solution
**EcoHydra** ingests local climate from **NASA POWER** and plant inputs (pressure, duration, filtration, oil type, control type) to estimate:
- **Efficiency (%)** and **waste heat (kWh)**
- **CO₂ (kg)** under two views: **constant input** vs **same output**
- **Business KPIs:** **$ saved/yr**, **Payback**, **NPV**, **IRR**
- **Sensitivity** to energy price & utilization (±20%)
- One-click export: **ExecutiveSummary.xlsx** (with assumptions + finance tabs)

It’s a Streamlit app with trained models saved in `./artifacts`.

---

## 3) NASA Open Data Used
- **NASA POWER** (Global meteorology & solar):  
  - Daily **2 m air temperature (T2M)**  
  - Daily **10 m wind speed (WS10M)**  
  API base: `https://power.larc.nasa.gov/api/`  
  Example endpoint used (daily, point):
POWER provides global coverage without an API key and anchors ambient/convective loss assumptions.

---

## 4) How It Works
1. **Location → NASA POWER:** user sets site (lat/long, year). App fetches T2M/WS10M.  
2. **Baseline & Upgrade:** user selects control type, filtration, oil type, pressure, duration, etc.  
3. **Models**  
 - `model_efficiency.joblib` → predicts **Efficiency %**  
 - `model_co2.joblib` → predicts **CO₂ (kg)** given energy & efficiency

https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,WS10M&start=20250101&end=20251231&latitude=30.27&longitude=-97.74&format=JSON


4. **Scenarios**  
 - **Constant input** (same kWh in)  
 - **Same output** (adjusted kWh for equal useful work)  
5. **Finance:** computes **$ saved/yr**, **Payback**, **NPV/IRR** (user-set price/discount rate).  
6. **Export:** `ExecutiveSummary.xlsx` with **Assumptions, Savings, Finance, Scenarios**.

**Architecture (logical):**  
`User → Streamlit UI → NASA POWER API → ML models (eff & CO₂) → Finance engine (NPV/IRR) → Excel export`

---

## 5) App Features
- **Scenario Comparison:** bars & table for CO₂ and waste energy across views.  
- **Business View:** KPIs (**$ saved**, **Payback**, **NPV/IRR**), plus a **sensitivity** mini-chart.  
- **Recommendations:** one-change-at-a-time impact (servo/proportional, filtration, oil, materials).  
- **Past Data:** histograms & scatter trend (requires `hydraulics_ops_sustainability_dataset.csv`).  
- **Research Paper:** link/embed your preprint (Zenodo/TechRxiv).  
- **Admin:** quick re-train and model metrics.

---

## 6) Quickstart

### Prereqs
- Python **3.10+**

### Install & run
```bash
# (recommended) create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# First-time: train & save artifacts
python hydraulics.py

# Launch app
streamlit run app.py
