# AI Use Log: HydroSense-Kenya Project

This log maintains a transparent record of all Generative AI tools leveraged during the scoping, mathematical modeling, code development, and report synthesis phases of the HydroSense-Kenya precision irrigation project.

## 1. Executive Summary of AI Integration

- **Primary Tool Used:** Google Gemini and Claude(Large Language Model)
- **Human Role:** Core software architecture design, system logic boundaries, physical parameter tuning, integration testing, and technical validation.
- **AI Role:** Boilerplate generation, syntax optimization, multi-panel rendering troubleshooting, and layout design for presentation slides and scientific reports.

---

## 2. Component Breakdown and Attribution

| Project Component | AI Contribution Level | Specific AI Tasks Completed | Human Review & Validation Action |
| :--- | :---: | :--- | :--- |
| **Data Preprocessing** | Low | Generated standard `pandas` aggregation code patterns (`.groupby().agg()`). | Inspected telemetry timestamps to ensure temporal alignment of data vectors. |
| **Physics Engine (`water_balance_step`)** | Medium | Assisted in structuring the Forward Euler discrete state equation. | Manually verified boundary conditions, including non-linear drainage above Field Capacity (40.0%). |
| **Optimization Engine** | Medium | Suggested structure for tracking iterations, learning rates, and gradient penalties. | Tuned learning rate ($\alpha = 0.04$) and established a strict penalty scalar ($\lambda = 150.0$). |
| **Visualization Suite** | High | Fixed subplot alignment issues and corrected spacing for missing bars in Panel D. | Implemented custom `np.nan_to_num` data cleaning to safeguard the rendering process. |
| **Reporting Artifacts** | High | Crafted clean HTML templates for the slide deck and WeasyPrint Python scripts for the final PDF. | Audited all technical metrics in the summary table to ensure exact numerical correspondence with simulated trajectories. |

---

## 3. Major Prompt Iterations & Engineering Notes

### Incident 1: Panel D Rendering Bug
- **Context:** The original multi-panel plot hid or failed to render the irrigation comparison bar chart due to notebook layout clipping.
- **Prompt Logic:** Provided the faulty code block and error states, requesting a hardened matplotlib layout wrapper.
- **Outcome:** Received advice on setting explicit x-limits, implementing `np.nan_to_num`, and swapping out `plt.tight_layout()` for manual `plt.subplots_adjust()` control.

### Incident 2: Notebook Session State Recovery
- **Context:** Hit a sudden `NameError` for `total_deficit_trap` and `res_bi` due to notebook memory clearing.
- **Prompt Logic:** Requested an inline calculation script to dynamically recalculate environmental stats via composite trapezoidal integration (`np.trapz`).
- **Outcome:** Successfully calculated the 92.45 mm moisture deficit on the fly, preventing runtime crashes.

---

## 4. Verification, Safety, and Quality Control Workflow

To ensure code safety, accuracy, and engineering integrity, a strict verification pipeline was enforced:

1. **Unit Testing Suite:** A total of 54 automated integration and unit testing checks were executed locally against code outputs. Every math module, physics transformation, and numerical method was fully evaluated across extreme edge conditions.
2. **Deterministic Validation:** AI-generated optimization trajectories were cross-checked directly against deterministic baselines (such as steady-state calculations solved independently via the Bisection root-finding engine).
3. **Data Boundary Enforcement:** State trajectories were audited to ensure compliance with physical limitations (e.g., preventing soil moisture percentages from dropping below 0.0% or rising above structural saturation constraints).

## 5. Affirmation

All algorithmic choices, scientific conclusions, and data evaluations inside the final deliverables remain the sole intellectual responsibility of the author. The AI was deployed purely as a technical pair-programmer to streamline implementation efficiency and ensure visual presentation compliance.