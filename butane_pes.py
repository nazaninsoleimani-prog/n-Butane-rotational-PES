"""
n-Butane rotational PES exercise
(from Chris Cramer, CompChem.02.01 / Chem 4021/8021)

1) Relaxed rotational reaction coordinate (C2-C3 dihedral)
2) Rigid scan with other DOF fixed at *trans* (anti) geometry
3) Rigid scan with other DOF fixed at *gauche* geometry
4) Conceptual difference: internal vs Cartesian PES topology

Energies from RDKit MMFF94 (educational molecular-mechanics model).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

# Carbon atoms in n-butane SMILES CCCC are indices 0,1,2,3
# Dihedral of interest: C1-C2-C3-C4  (atoms 0-1-2-3)


def make_butane():
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def mmff_energy(mol) -> float:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    return ff.CalcEnergy()  # kcal/mol


def set_dihedral(mol, angle_deg: float):
    """Set C-C-C-C dihedral (atoms 0-1-2-3) in degrees."""
    conf = mol.GetConformer()
    Chem.rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, float(angle_deg))


def get_dihedral(mol) -> float:
    conf = mol.GetConformer()
    return Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)


def optimize_with_fixed_dihedral(mol, angle_deg: float, max_its: int = 500):
    """Relaxed scan: fix dihedral, minimize all other degrees of freedom."""
    m = Chem.Mol(mol)
    set_dihedral(m, angle_deg)
    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    # restrain the dihedral near the target angle
    ff.MMFFAddTorsionConstraint(0, 1, 2, 3, False, float(angle_deg), float(angle_deg), 1.0e5)
    ff.Minimize(maxIts=max_its)
    return m, ff.CalcEnergy()


def rigid_energy_from_reference(ref_mol, angle_deg: float) -> float:
    """
    Rigid scan: take a reference geometry (trans or gauche optimized),
    only change the C2-C3 dihedral; do not reoptimize other DOF.
    """
    m = Chem.Mol(ref_mol)
    set_dihedral(m, angle_deg)
    return mmff_energy(m)


def find_local_minima(angles, energies, n_local=3):
    """Return angles of lowest local minima (for labeling)."""
    e = np.asarray(energies)
    a = np.asarray(angles)
    # shift so min is 0 for comparison
    idx = []
    for i in range(1, len(e) - 1):
        if e[i] <= e[i - 1] and e[i] <= e[i + 1]:
            idx.append(i)
    # also check endpoints loosely
    order = sorted(idx, key=lambda i: e[i])
    return [(a[i], e[i]) for i in order[:n_local]]


def run_scan(step=5.0):
    angles = np.arange(-180.0, 180.0 + 0.1, step)

    base = make_butane()

    # --- 1) Relaxed scan ---
    relaxed_E = []
    relaxed_mols = []
    for ang in angles:
        m, e = optimize_with_fixed_dihedral(base, ang)
        relaxed_E.append(e)
        relaxed_mols.append(m)
    relaxed_E = np.array(relaxed_E)
    relaxed_E -= relaxed_E.min()

    # reference structures near expected minima
    # anti ~ ±180 or 180; gauche ~ ±60
    # pick best relaxed structures near those angles
    def nearest_mol(target):
        i = int(np.argmin(np.abs(angles - target)))
        return Chem.Mol(relaxed_mols[i])

    anti_ref = nearest_mol(180.0)
    # normalize anti to exactly 180 for rigid scan
    set_dihedral(anti_ref, 180.0)
    # re-minimize with dihedral fixed at 180
    anti_ref, _ = optimize_with_fixed_dihedral(base, 180.0)

    gauche_ref, _ = optimize_with_fixed_dihedral(base, 60.0)

    # --- 2) Rigid from trans (anti) ---
    rigid_anti_E = np.array([rigid_energy_from_reference(anti_ref, ang) for ang in angles])
    rigid_anti_E -= rigid_anti_E.min()

    # --- 3) Rigid from gauche ---
    rigid_gauche_E = np.array([rigid_energy_from_reference(gauche_ref, ang) for ang in angles])
    rigid_gauche_E -= rigid_gauche_E.min()

    return angles, relaxed_E, rigid_anti_E, rigid_gauche_E


def plot_results(angles, relaxed_E, rigid_anti_E, rigid_gauche_E, save=True, show=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles, relaxed_E, "k-", linewidth=2.2, label="1) Relaxed scan (all other DOF optimized)")
    ax.plot(
        angles,
        rigid_anti_E,
        "b--",
        linewidth=1.6,
        label="2) Rigid scan (geometry frozen at anti/trans)",
    )
    ax.plot(
        angles,
        rigid_gauche_E,
        "r:",
        linewidth=1.8,
        label="3) Rigid scan (geometry frozen at gauche)",
    )

    # Labels for classic features on the *relaxed* curve
    labels = {
        180: "anti (trans)\nminimum",
        -180: None,  # same as 180 for periodic plot ends
        60: "gauche\nminimum",
        -60: "gauche\nminimum",
        0: "syn / fully\neclipsed (max)",
        120: "eclipsed\nbarrier",
        -120: "eclipsed\nbarrier",
    }
    for ang, text in labels.items():
        if text is None:
            continue
        # find nearest computed point
        i = int(np.argmin(np.abs(angles - ang)))
        ax.annotate(
            text,
            xy=(angles[i], relaxed_E[i]),
            xytext=(0, 12 if "min" in text else 18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="darkgreen" if "min" in text else "darkred",
        )
        ax.plot(angles[i], relaxed_E[i], "ko", markersize=4)

    ax.set_xlabel(r"C1–C2–C3–C4 dihedral angle $\phi$ / degrees")
    ax.set_ylabel("Relative energy / kcal mol$^{-1}$ (MMFF94)")
    ax.set_title("n-Butane rotational reaction coordinate\n(Cramer CompChem.02.01 exercise)")
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if save:
        out = RESULTS / "butane_rotational_pes.png"
        fig.savefig(out, dpi=160)
        print("Saved:", out)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_answers(angles, relaxed_E, rigid_anti_E, rigid_gauche_E):
    print("=" * 60)
    print("n-BUTANE ROTATION — EXERCISE ANSWERS")
    print("=" * 60)
    print(
        """
1) RELAXED reaction coordinate
   - Reaction coordinate = C2–C3 dihedral angle phi of n-butane.
   - At each phi, ALL other bond lengths/angles/dihedrals are optimized.
   - Classic shape (relative energy):
       * Global minima near phi = ±180 deg  →  anti (trans) conformer
       * Local minima near phi = ±60 deg    →  gauche conformers
         (gauche is a few kcal/mol higher than anti)
       * Highest barrier near phi = 0 deg   →  syn fully eclipsed
       * Lower barriers near phi = ±120 deg →  eclipsed H/Me interactions
   - This is the true 1-D *reaction coordinate* slice of the PES.

2) RIGID scan with DOF fixed at TRANS (anti) geometry
   - Only phi is changed; bond lengths/angles stay at anti values.
   - Barriers are usually HIGHER than the relaxed scan, because
     the molecule cannot open angles / stretch bonds to relieve
     steric clash when groups eclipse.
   - Blue dashed curve on the plot.

3) RIGID scan with DOF fixed at GAUCHE geometry
   - Same idea, but frozen structure is the gauche minimum.
   - Curve is shifted/distorted relative to the anti-rigid scan:
     the energy at phi ≈ 60 deg is low (by construction), but
     near anti and other angles the energy can look worse because
     the frozen gauche bond angles are not ideal for those regions.
   - Red dotted curve on the plot.

4) Internal coordinates vs Cartesian coordinates (topology)
   - Cartesian PES: 3N coordinates (x,y,z for each atom). Energy is
     invariant to overall translation/rotation, so the surface has
     flat directions (null modes) and the same chemical structure
     appears infinitely many times (translated/rotated copies).
   - Internal coordinates (bonds, angles, torsions): typically 3N-6
     dimensions for a nonlinear molecule. They describe *shape*
     only, so each chemically distinct geometry appears once
     (no free translation/rotation).
   - Topology difference: internals are often curvilinear and can be
     periodic (dihedrals 0–360° wrap around) or singular at linear
     angles; Cartesians are flat Euclidean space but include
     redundant rigid-body degrees of freedom.
   - Chemistry cares about relative internal geometry; that is why
     reaction coordinates are usually internal (e.g. a torsion).
"""
    )
    # Numerical highlights from MMFF scan
    i_anti = int(np.argmin(np.abs(angles - 180)))
    i_g = int(np.argmin(np.abs(angles - 60)))
    i_0 = int(np.argmin(np.abs(angles - 0)))
    i_120 = int(np.argmin(np.abs(angles - 120)))
    print("MMFF94 numerical highlights (relaxed, kcal/mol relative to anti):")
    print(f"  anti  (~180 deg): {relaxed_E[i_anti]:.3f}")
    print(f"  gauche (~60 deg): {relaxed_E[i_g]:.3f}")
    print(f"  syn    (~0 deg):  {relaxed_E[i_0]:.3f}")
    print(f"  eclipsed (~120):  {relaxed_E[i_120]:.3f}")
    print(f"  rigid-anti barrier at 0 deg:   {rigid_anti_E[i_0]:.3f}")
    print(f"  rigid-gauche energy at 0 deg:  {rigid_gauche_E[i_0]:.3f}")


def main():
    print("Running n-butane MMFF scans (this may take ~30-60 s)...")
    angles, relaxed_E, rigid_anti_E, rigid_gauche_E = run_scan(step=5.0)
    print_answers(angles, relaxed_E, rigid_anti_E, rigid_gauche_E)
    plot_results(angles, relaxed_E, rigid_anti_E, rigid_gauche_E, save=True, show=False)

    # also save CSV
    import pandas as pd

    df = pd.DataFrame(
        {
            "dihedral_deg": angles,
            "E_relaxed_kcal": relaxed_E,
            "E_rigid_anti_kcal": rigid_anti_E,
            "E_rigid_gauche_kcal": rigid_gauche_E,
        }
    )
    csv_path = RESULTS / "butane_pes_scan.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
