# 🌀 n-Butane Rotational PES Exercise

Solution to the **n-butane rotation** exercise from  
**Chris Cramer – CompChem.02.01: The Potential Energy Surface**  
(University of Minnesota Chem 4021/8021)

🎬 Video: [CompChem.02.01 The Potential Energy Surface](https://www.youtube.com/watch?v=pu4uL7deCNw)

**Author:** Nazanin Soleimani  
**GitHub:** [nazaninsoleimani-prog](https://github.com/nazaninsoleimani-prog)

## 📝 The exercise (from the lecture slides)

**n-Butane rotation** — an elementary reaction coordinate:

1. ✏️ Draw and carefully label a **relaxed** butane rotational reaction coordinate  
2. 📐 Overlay the energy if all other DOF are **fixed at the trans (anti)** geometry  
3. 📐 Overlay the same for the **gauche** geometry  
4. 💭 How is a PES in **internal coordinates** different from one in **Cartesian coordinates**?

## ✨ What this project does
- 🧬 Build n-butane with **RDKit**
- ⚡ Evaluate energies with **MMFF94** force field
- 📈 Compute a **relaxed** C2–C3 dihedral scan
- 🔒 Compute **rigid** scans frozen at anti and gauche
- 🖼️ Plot all three curves with labels (anti, gauche, eclipsed barriers)
- 💾 Export CSV + PNG results

## 📂 Project structure
- 📓 `Butane_PES_Exercise.ipynb` — step-by-step notebook
- 🐍 `butane_pes.py` — full analysis script
- 📤 `results/` — plot + CSV
- 📦 `requirements.txt`
- 📄 `README.md`

## ▶️ How to run

```bash
cd "C:\Users\nazan\Documents\Pyton Projects\Butane-PES-Exercise"
pip install -r requirements.txt
python butane_pes.py
python -m notebook
```

Open `Butane_PES_Exercise.ipynb` and run cells top → bottom.

## 📊 Expected chemistry (relaxed scan)

| Feature | Approx. dihedral φ | Role |
|---------|---------------------|------|
| 🟢 anti (trans) | ±180° | global minimum |
| 🟡 gauche | ±60° | local minimum |
| 🔴 eclipsed | ±120° | barrier |
| 🔴 syn eclipsed | 0° | highest barrier |

Rigid scans usually show **higher barriers** than the relaxed scan (the molecule cannot relax steric strain).

## 🛠️ Requirements
- 🐍 Python 3
- 🧬 RDKit
- 🔢 numpy, pandas, matplotlib
- 📓 Jupyter (optional)

## 📚 Reference
- Cramer, C. J. — Chem 4021/8021 Computational Chemistry  
- Lecture slides: *Preamble to the Basic Force Field* (Video II.i)

## 👩‍💻 Author
**Nazanin Soleimani** 🌟

---

💡 Feel free to improve this further anytime! 🚀
