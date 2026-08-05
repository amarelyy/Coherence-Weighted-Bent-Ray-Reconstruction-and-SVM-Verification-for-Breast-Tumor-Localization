"""
visualize_results.py — Generate visualizations from ablation results.
Usage: python visualize_results.py --results-dir results
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

def load_all_results(results_dir):
    results_dir = Path(results_dir)
    csvs = sorted([f for f in results_dir.glob("ablation_*.csv") if "summary" not in f.name])
    if not csvs:
        print(f"No ablation_*.csv in {results_dir}"); return None
    dfs = []
    for cp in csvs:
        df = pd.read_csv(cp)
        df["variant"] = cp.stem.replace("ablation_", "").replace("_", "+")
        dfs.append(df)
        vn = cp.stem.replace("ablation_", "").replace("_", "+")
        print(f"  {vn}: {len(df)} scans, LE={df['localization_error_mm'].mean():.1f}mm")
    return pd.concat(dfs, ignore_index=True)

def _color(v):
    if "GIBR" in v: return "#e74c3c"
    if "CF" in v: return "#2ecc71"
    return "#3498db"

def plot_variant_comparison(df, out):
    s = df.groupby("variant").agg(
        mean_le=("localization_error_mm","mean"), std_le=("localization_error_mm","std"),
        det=("localization_error_mm", lambda x: (x<=20).mean()),
    ).reset_index().sort_values("mean_le")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    c=[_color(v) for v in s["variant"]]
    a1.barh(range(len(s)),s["mean_le"],xerr=s["std_le"],color=c,alpha=.8,capsize=3)
    a1.set_yticks(range(len(s))); a1.set_yticklabels(s["variant"])
    a1.set_xlabel("Mean LE (mm)"); a1.set_title("Localization Error by Variant")
    a1.axvline(20,color="red",ls="--",alpha=.5,label="20mm threshold"); a1.legend()
    for i,(_,r) in enumerate(s.iterrows()): a1.text(r["mean_le"]+r["std_le"]+1,i,f"{r['mean_le']:.1f}",va="center",fontsize=9)
    a2.barh(range(len(s)),s["det"]*100,color=c,alpha=.8)
    a2.set_yticks(range(len(s))); a2.set_yticklabels(s["variant"])
    a2.set_xlabel("Detection @20mm (%)"); a2.set_title("Detection Rate by Variant")
    for i,(_,r) in enumerate(s.iterrows()): a2.text(r["det"]*100+1,i,f"{r['det']:.0%}",va="center",fontsize=9)
    plt.tight_layout(); plt.savefig(out/"variant_comparison.png",bbox_inches="tight"); plt.close()
    print(f"  Saved: variant_comparison.png")

def plot_le_boxplot(df, out):
    vs=sorted(df["variant"].unique())
    data=[df[df["variant"]==v]["localization_error_mm"].values for v in vs]
    fig,ax=plt.subplots(figsize=(12,6))
    bp=ax.boxplot(data,patch_artist=True,widths=.6)
    ax.set_xticklabels(vs)
    for p,c in zip(bp["boxes"],[_color(v) for v in vs]): p.set_facecolor(c); p.set_alpha(.6)
    ax.axhline(20,color="red",ls="--",alpha=.5); ax.set_ylabel("LE (mm)")
    ax.set_title("LE Distribution by Variant"); plt.xticks(rotation=30,ha="right")
    plt.tight_layout(); plt.savefig(out/"le_distribution.png",bbox_inches="tight"); plt.close()
    print(f"  Saved: le_distribution.png")
    
def plot_per_phantom_heatmap(df, out):
    piv=df.pivot_table(values="localization_error_mm",index="phant_id",columns="variant",aggfunc="mean")
    def sk(pid):
        m=re.match(r"A(\d+)F(\d+)",str(pid))
        return (int(m.group(1)),int(m.group(2))) if m else (999,999)
    piv=piv.loc[sorted(piv.index,key=sk)]
    fig,ax=plt.subplots(figsize=(12,max(8,len(piv)*.35)))
    im=ax.imshow(piv.values,cmap="RdYlGn_r",aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,rotation=45,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=8)
    ax.set_title("Mean LE (mm) per Phantom per Variant")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=piv.values[i,j]
            if not np.isnan(v): ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6,color="white" if v>40 else "black")
    plt.colorbar(im,ax=ax,label="LE (mm)",shrink=.8)
    plt.tight_layout(); plt.savefig(out/"per_phantom_heatmap.png",bbox_inches="tight"); plt.close()
    print(f"  Saved: per_phantom_heatmap.png")

def plot_birads(df, out):
    if "birads" not in df.columns: return
    fig,ax=plt.subplots(figsize=(10,6))
    for v in sorted(df["variant"].unique()):
        sub=df[df["variant"]==v]; bm=sub.groupby("birads")["localization_error_mm"].mean()
        ax.plot(bm.index,bm.values,marker="o",label=v,lw=2)
    ax.set_xlabel("BI-RADS Density"); ax.set_ylabel("Mean LE (mm)")
    ax.set_title("LE by Breast Density"); ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(["C1 Fatty","C2","C3","C4 Dense"]); ax.axhline(20,color="red",ls="--",alpha=.3)
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out/"birads_stratified.png",bbox_inches="tight"); plt.close()
    print(f"  Saved: birads_stratified.png")

def plot_feature_scatter(df, out):
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,(xc,xl) in zip(axes,[("scr_db","SCR (dB)"),("blob_compactness","Compactness"),("breast_radius_mm","Breast Radius (mm)")]):
        if xc not in df.columns: continue
        for v in sorted(df["variant"].unique()):
            sub=df[df["variant"]==v]
            ax.scatter(sub[xc],sub["localization_error_mm"],alpha=.3,s=10,label=v)
        ax.set_xlabel(xl); ax.set_ylabel("LE (mm)"); ax.axhline(20,color="red",ls="--",alpha=.3)
        ax.set_title(f"{xl} vs LE"); ax.legend(fontsize=6,markerscale=3)
    plt.tight_layout(); plt.savefig(out/"feature_scatter.png",bbox_inches="tight"); plt.close()
    print(f"  Saved: feature_scatter.png")
    
def print_summary(df, out):
    s=df.groupby("variant").agg(
        n=("localization_error_mm","count"), mean_le=("localization_error_mm","mean"),
        med_le=("localization_error_mm","median"), det20=("localization_error_mm",lambda x:(x<=20).mean()),
        iou=("iou","mean"), dice=("dice","mean"), scr=("scr_db","mean"),
    ).reset_index().sort_values("mean_le")
    s.to_csv(out/"summary_table.csv",index=False,float_format="%.3f")
    print(f"\n{'Variant':<18}{'N':>5}{'MeanLE':>10}{'MedLE':>10}{'Det@20':>8}{'IoU':>8}{'Dice':>8}{'SCR':>8}")
    print("-"*75)
    for _,r in s.iterrows():
        print(f"{r['variant']:<18}{int(r['n']):>5}{r['mean_le']:>8.1f}mm{r['med_le']:>8.1f}mm{r['det20']:>7.0%}{r['iou']:>8.4f}{r['dice']:>8.4f}{r['scr']:>7.2f}")
    print(f"\nSaved: summary_table.csv")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",default="results")
    p.add_argument("--output-dir",default=None)
    a=p.parse_args()
    rd=Path(a.results_dir)
    od=Path(a.output_dir) if a.output_dir else rd/"figures"
    od.mkdir(parents=True,exist_ok=True)
    df=load_all_results(rd)
    if df is None: return
    print(f"\nGenerating figures ({len(df)} rows)...")
    plot_variant_comparison(df,od)
    plot_le_boxplot(df,od)
    plot_per_phantom_heatmap(df,od)
    plot_birads(df,od)
    plot_feature_scatter(df,od)
    print_summary(df,od)
    print(f"\nAll saved to: {od}/")

if __name__=="__main__":
    main()