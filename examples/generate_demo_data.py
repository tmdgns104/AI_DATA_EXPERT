from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)


def manufacturing(rows: int = 2000) -> pd.DataFrame:
    line = RNG.choice(["Line-A", "Line-B"], rows)
    material = RNG.choice(["Type-X", "Type-Y", "Type-Z"], rows, p=[0.45, 0.35, 0.20])
    temp = RNG.normal(100, 4.5, rows)
    pressure = RNG.normal(50, 3.2, rows)
    humidity = RNG.normal(55, 7, rows)
    vibration = np.clip(RNG.normal(0.8, 0.28, rows), 0.05, None)
    processing = RNG.normal(120, 13, rows)
    power = RNG.normal(220, 25, rows)
    experience = RNG.integers(0, 16, rows).astype(float)

    material_effect = pd.Series(material).map({"Type-X": 0.6, "Type-Y": 0.0, "Type-Z": -0.8}).to_numpy()
    line_effect = pd.Series(line).map({"Line-A": 0.15, "Line-B": -0.1}).to_numpy()
    yield_pct = (
        94.2
        + material_effect
        + line_effect
        + 0.07 * experience
        - 0.045 * np.abs(temp - 100)
        - 0.025 * np.abs(pressure - 50)
        - 0.010 * np.abs(processing - 120)
        + RNG.normal(0, 0.48, rows)
    )

    df = pd.DataFrame({
        "production_line": line,
        "material_type": material,
        "temp_sensor_1": temp.round(2),
        "pressure_sensor": pressure.round(2),
        "humidity": humidity.round(2),
        "vibration_sensor": vibration.round(4),
        "processing_time_sec": processing.round(2),
        "power_consumption": power.round(2),
        "operator_experience_years": experience,
        "yield_percentage": yield_pct.round(2),
    })

    for col, rate in {"temp_sensor_1": 0.025, "pressure_sensor": 0.035, "humidity": 0.03, "processing_time_sec": 0.03}.items():
        idx = RNG.choice(df.index, int(rows * rate), replace=False)
        df.loc[idx, col] = np.nan
    return df


def classification(rows: int = 1200) -> pd.DataFrame:
    a = RNG.normal(size=rows)
    b = RNG.normal(size=rows)
    line = RNG.choice(["A", "B", "C"], rows)
    score = 1.2 * a - 0.8 * b + (line == "C") * 0.5 + RNG.normal(0, 0.7, rows)
    threshold = np.quantile(score, 0.92)
    defect = np.where(score >= threshold, "defect", "normal")
    return pd.DataFrame({"sensor_a": a, "sensor_b": b, "line": line, "defect": defect})


def main() -> None:
    m = ROOT / "4_manufacturing_yield.csv"
    c = ROOT / "classification_example.csv"
    if not m.exists():
        manufacturing().to_csv(m, index=False)
        print(f"created {m}")
    if not c.exists():
        classification().to_csv(c, index=False)
        print(f"created {c}")


if __name__ == "__main__":
    main()
