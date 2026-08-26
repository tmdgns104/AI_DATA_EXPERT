from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

REQUIRED_MODULES = [
    "numpy", "pandas", "scipy", "sklearn", "statsmodels",
    "matplotlib", "nbformat", "nbclient", "ipykernel",
    "torch", "torchvision", "PIL", "yaml",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append({"module": module, "error": f"{type(exc).__name__}: {exc}"})

    if missing:
        print(json.dumps({"status": "FAIL", "stage": "dependencies", "missing": missing}, ensure_ascii=False, indent=2))
        return 1

    skill = root / ".agents" / "skills" / "ai-data-expert" / "SKILL.md"
    if not skill.exists():
        print(json.dumps({"status": "FAIL", "stage": "skill", "error": str(skill)}, ensure_ascii=False, indent=2))
        return 1

    sys.path.insert(0, str(root / "data_expert"))
    try:
        from enhanced_system import EnhancedSystem
        import numpy as np
        import pandas as pd
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "stage": "runtime_import", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 1

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "smoke.csv"
        rng = np.random.default_rng(42)
        n = 120
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        target = (x1 + 0.4 * x2 > 0).astype(int)
        pd.DataFrame({"sensor_a": x1, "sensor_b": x2, "status": target}).to_csv(path, index=False)
        problem = {
            "id": "INSTALL-SMOKE",
            "task": "train a classification model to predict status",
            "data_path": str(path),
            "profile": {
                "modality": "tabular",
                "target": "status",
                "target_type": "categorical",
                "rows": n,
                "prediction_time": "after sensors are measured and before decision",
                "business_cost": {"false_positive": 1, "false_negative": 2},
            },
        }
        try:
            result = EnhancedSystem().run(problem)
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "stage": "runtime_smoke", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
            return 1

    route = result.get("routing", {}).get("execution_order", [])
    verification = result.get("verification", {}).get("status")
    errors = result.get("expert_errors", [])
    ok = "machine-learning" in route and verification != "FAIL" and not errors
    payload = {
        "status": "PASS" if ok else "FAIL",
        "python": sys.version.split()[0],
        "skill": str(skill.relative_to(root)),
        "route": route,
        "verification": verification,
        "expert_errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
