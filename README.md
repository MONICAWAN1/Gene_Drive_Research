# Gene Drive project

Repository for simulation models

## Top-Level Directories

### [`analysis/`](/analysis)
Files that get simulation results, run stability analysis, mapping, and plotting figures

### [`models/`](/models)
Gene drive and non-gene-drive models

### [`scripts/`](/scripts)
Scripts to run files in analysis for mapping or plotting results

### [`utils/`](/utils)
Helper functions


## 🧪 Typical Workflow

1. **Generate simulation curves**

    ```bash
    python analysis/get_curves.py
    ```

    *Inside `get_curves.py` you can change*  
    `h` – dominance coefficient  
    `model_type` – which GD/NGD model to run  
    `step` – integration step (e.g. 0.01 or 0.1)

2. **Run stability analysis**

    ```bash
    python analysis/stability.py
    ```

    Edit `main()` (or the `runall` helper) to set the parameter grid for  
    `s`, `c`, `h` → selection, conversion rate, dominance.

3. **Create mapping data**

    ```bash
    python analysis/run_mapping.py <map_function> <h> <gdFile> <-s>
    ```

    *Tips*  
    • Example GD results file: `h{currH}_allgdRes001G`  
    • `gdFile` is either `001` or `01`
    • If called with command `-s`, mapped results are saved
    • Adjust regime conditions & output filenames inside `run_mapping.py`.

4. **Plot results**

    ```bash
    python scripts/run_plot.py
    ```

---

## ⚙️ Mapping & Comparison Pipeline

This project focuses on mapping GD configurations ( `s, c, h` ) to NGD parameters ( `s_ngd, h_ngd` ) that best reproduce allele‑frequency trajectories.

* **Fixation / Loss regimes** ― cases where the GD allele fixes or is lost.  
* **Stable‑equilibrium regimes** ― interior equilibria are analyzed for stability.  
* **Error comparison** ― `getDiff.py` computes mean‑squared‑error (MSE) gaps across regimes.

---

## 🛠️ Requirements

* Python ≥ 3.7  
* Install with:

    ```bash
    pip install -r requirements.txt
    ```

**Core libraries**

| Library | Purpose               |
|---------|-----------------------|
| NumPy   | numerical routines    |
| SciPy   | ODE integration       |
| pandas  | data handling         |
| matplotlib | plotting           |

---

## 📊 Outputs

* Time‑series allele‑frequency curves  
* Phase and stability diagrams  
* Mapping‑error heatmaps & scatter plots  

---

## 🧬 About Gene Drives

Gene drives are genetic mechanisms that bias inheritance, enabling a chosen allele to spread through a population faster than Mendelian rules predict. They hold promise for controlling vectors (e.g. malaria‑carrying mosquitoes), invasive species, and agricultural pests—but raise ecological and ethical questions. 

---

## 📫 Contact

Maintained by **[Monica Wan](https://github.com/MONICAWAN1)**.