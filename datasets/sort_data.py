from pathlib import Path
import shutil

base_dir = Path(__file__).resolve().parent

root = base_dir / "SIDD_Small_sRGB_Only" / "Data"
target = base_dir / "SIDD_small"

print(f"Data: {root}, Target: {target}")

(target / "noisy").mkdir(parents=True, exist_ok=True)
(target / "clean").mkdir(parents=True, exist_ok=True)

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue

    noisy = list(folder.glob("NOISY_*.PNG"))
    clean = list(folder.glob("GT_*.PNG"))

    if noisy and clean:
        base_name = f"{folder.name}.png"
        shutil.copy(noisy[0], target / "noisy" / base_name)
        shutil.copy(clean[0], target / "clean" / base_name)
        print(f"Copied {base_name}")
