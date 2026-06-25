# HydroSense-Kenya

**ICS 2207 Scientific Computing — Capstone Project**

A scientific computing system for smart irrigation, water balance simulation, and climate-aware decision support for Kenyan smallholder farms.

---

## Project Overview

HydroSense-Kenya models soil-water balance, estimates evapotranspiration, simulates future soil-moisture trajectories, and recommends optimized irrigation schedules using data from three farm zones (Zone_A: tomato, Zone_B: kale, Zone_C: maize).

### Central Scientific Question
> Given weather and soil-sensor data, how can we model water availability, estimate water deficit, simulate future soil moisture, and recommend an efficient irrigation plan that minimizes water use without exposing crops to moisture stress?

---

## Repository Structure

```
HydroSense-Kenya/
├── data/
│   ├── raw/                        
│   │   ├── weather_daily.csv
│   │   ├── soil_sensor_data.csv
│   │   └── crop_zone_parameters.csv
│   └── processed/
│       └── cleaned_irrigation_dataset.csv
├── notebooks/
│   ├── Level_1_Problem_Framing.ipynb
│   ├── Level_2_Vectorization_and_Error.ipynb
│   ├── Level_3_Numerical_Methods.ipynb
│   ├── Level_4_Data_Analysis_and_Visualization.ipynb
│   ├── Level_5_Simulation_and_Optimization.ipynb
│   └── Level_6_Final_Integration.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── numerical_methods.py
│   ├── simulation.py
│   ├── optimization.py
│   └── visualization.py
├── tests/
│   ├── test_root_finding.py
│   ├── test_integration.py
│   ├── test_linear_systems.py
│   └── test_simulation.py
├── reports/
│   ├── final_scientific_report.pdf 
│   └── presentation_slides.pdf       
├── AI_USE_LOG.md
├── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd HydroSense-Kenya

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          

# 3. Install dependencies
pip install 

# 4. Place raw datasets in data/raw/
#    weather_daily.csv, soil_sensor_data.csv,  crop_zone_parameters.csv

# 5. Launch JupyterLab
jupyter lab
```

---

## Core Water-Balance Model

```
S(t+1) = S(t) + R(t) + I(t) - ET(t) - D(t)
```

| Symbol | Meaning |
|--------|---------|
| S_t | Soil water storage (% vol) at time t |
| R_t | Effective rainfall contribution |
| I_t | Irrigation applied |
| ET_t | Evapotranspiration (crop water loss) |
| D_t | Drainage beyond field capacity |

### Evapotranspiration Formula (simplified)

```
ET = max(0,  0.12·T  +  0.35·W  +  2.4·Solar  −  0.025·H)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Six-Level Project Milestones

| Level | Focus | Marks |
|-------|-------|-------|
| 1 | Problem framing, Python foundations, data dictionary | 10 |
| 2 | NumPy vectorization, floating-point error, propagation | 15 |
| 3 | Root finding, differentiation, integration, linear systems | 20 |
| 4 | Pandas data cleaning, statistical summaries, visualization | 15 |
| 5 | Simulation (Euler/RK4), Monte Carlo, optimization | 25 |
| 6 | AI-use validation, testing, reproducibility, presentation | 15 |

---

- Winnie Mugoiri
- Course: ICS 2207 Scientific Computing
- Semester: February–May 2026
