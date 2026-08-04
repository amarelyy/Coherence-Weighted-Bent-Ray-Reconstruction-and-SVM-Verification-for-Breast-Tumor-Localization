import trimesh, numpy as np, json
from pathlib import Path

data_dir = Path("data")
fibro_files = sorted(data_dir.glob("F*.stl"))
adipose_files = sorted(data_dir.glob("A*.stl"))
print(f"Fibro: {[f.name for f in fibro_files]}")
print(f"Adipose: {[f.name for f in adipose_files]}\n")

def get_cs(mesh, z_frac):
    z = mesh.bounds[0][2] + z_frac * (mesh.bounds[1][2] - mesh.bounds[0][2])
    sec = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
    if sec is None: return None
    v = sec.vertices
    x, y = v[:,0], v[:,1]
    cx, cy = np.mean(x), np.mean(y)
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    return dict(z=float(z), r_mean=float(np.mean(r)), r_min=float(np.min(r)),
                r_max=float(np.max(r)), centroid=[float(cx),float(cy)])

print("="*70)
print("GAP: Fibro vs Adipose at z=80%")
print("="*70)
results = []
for ff in fibro_files:
    num = ff.stem.replace("F","")
    af = data_dir / f"A{num}.stl"
    if not af.exists():
        print(f"  {ff.stem}: no A{num}.stl"); continue
    fm = trimesh.load(str(ff), force="mesh")
    am = trimesh.load(str(af), force="mesh")
    fs = get_cs(fm, 0.80)
    as_ = get_cs(am, 0.80)
    if fs and as_:
        gap = as_["r_mean"] - fs["r_mean"]
        gap_min = as_["r_min"] - fs["r_max"]
        results.append(dict(pair=f"F{num}+A{num}", fibro_r=fs["r_mean"],
                           adipose_r=as_["r_mean"], gap=gap, gap_min=gap_min))
        print(f"  F{num}+A{num}: fibro={fs['r_mean']:.1f}mm adipose={as_['r_mean']:.1f}mm gap={gap:.1f}mm gap_min={gap_min:.1f}mm")

if results:
    gaps = [r["gap"] for r in results]
    print(f"\nMean gap: {np.mean(gaps):.1f} +/- {np.std(gaps):.1f} mm")
    print(f"Min gap:  {np.min(gaps):.1f} mm")
    print(f"Max gap:  {np.max(gaps):.1f} mm")

Path("data/boundaries").mkdir(exist_ok=True)
with open("data/boundaries/gap_analysis.json","w") as f:
    json.dump(results, f, indent=2)
print("Saved: data/boundaries/gap_analysis.json")