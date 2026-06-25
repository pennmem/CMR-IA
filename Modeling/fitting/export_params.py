"""
Export best-fit parameters from an archived PSO run to the corresponding
Analysis data folder as a JSON file.

Usage:
    python export_params.py <archive_name> <simu_name>

Example:
    python export_params.py simu8_run1 8
"""

import sys
import json
import numpy as np
from pathlib import Path
from CMR_IA.fitting import make_boundary


SIMU_TO_DIR = {
    "1":     "simu1_recog_recsim",
    "2":     "simu2_recog_conti",
    "2b":    "simu2b_recog_assoc_conti",
    "3":     "simu3_recog_forget",
    "4":     "simu4_recog_wfe",
    "4base": "simu4_recog_wfe",
    "4shift":"simu4_recog_wfe",
    "4attn": "simu4_recog_wfe",
    "5":     "simu5_cr_rec",
    "6a":    "simu6a_cr_recsym",
    "6b":    "simu6b_cr_sym",
    "7":     "simu7_cr_pliili",
    "8":     "simu8_cr_sim",
    "S1":    "simuS1_recog_cr",
    "S2":    "simuS2_recog_recog",
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    archive_name = sys.argv[1]
    simu_name = sys.argv[2]

    if simu_name not in SIMU_TO_DIR:
        raise ValueError(f"Unknown simu_name: {simu_name!r}")

    script_dir = Path(__file__).parent
    xoptb_path = script_dir / "past_fits" / archive_name / "outfiles" / "xoptb.txt"
    if not xoptb_path.exists():
        raise FileNotFoundError(f"xoptb.txt not found at {xoptb_path}")

    xopt = np.loadtxt(xoptb_path)
    _, _, what_to_fit = make_boundary(simu_name)

    if len(xopt) != len(what_to_fit):
        raise ValueError(
            f"Parameter count mismatch: xoptb.txt has {len(xopt)} values "
            f"but simu {simu_name!r} expects {len(what_to_fit)}"
        )

    params = {name: float(val) for name, val in zip(what_to_fit, xopt)}

    anal_dir = script_dir / ".." / ".." / "Analysis" / SIMU_TO_DIR[simu_name] / "data"
    out_path = anal_dir / f"{archive_name}.json"

    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved {len(params)} parameters to {out_path}")
