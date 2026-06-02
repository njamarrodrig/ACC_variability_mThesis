"""
Corrélation r(T_Drake, Δη_NB−SB NEMO)  —  1991–2023
Anomalies désaisonnalisées lissées 12 mois  —  lag 0 uniquement
══════════════════════════════════════════════════════════════════════════
Entrées :
  Transport : ./Transport_sv/drake_transport_1991_2023.nc
  Δη NB−SB  : ./outputs_SBNB_proxytransp/delta_eta_SBNB_NEMO_1991_2023.nc

Sorties dans ./outputs_corr_SBNB/
  corr_SBNB_timeseries.png   — séries temporelles superposées (double axe Y)
  corr_SBNB_scatter.png      — nuage de points + droite de régression
  corr_SBNB_results.txt      — statistiques récapitulatives
══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────

SMOOTH  = 12   # lissage en mois

DIR_OUT = Path("./outputs_corr_SBNB")
DIR_OUT.mkdir(parents=True, exist_ok=True)

FILE_T    = Path("./Transport_sv/drake_transport_1991_2023.nc")
FILE_DETA = Path("./outputs_SBNB_proxytransp/delta_eta_SBNB_NEMO_1991_2023.nc")

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT + ALIGNEMENT
# ─────────────────────────────────────────────────────────────────

print("Chargement…")
ds_t    = xr.open_dataset(FILE_T)
ds_deta = xr.open_dataset(FILE_DETA)

transport_xa = ds_t["drake_transport"]
t_T = pd.DatetimeIndex(transport_xa["time_counter"].values)
transport = pd.Series(transport_xa.values,
                      index=pd.PeriodIndex(t_T, freq="M"), name="T_Drake")

var_deta = [v for v in ds_deta.data_vars
            if v.lower() in ("zos", "delta_eta", "delta_zos",
                             "__xarray_dataarray_variable__")]
if not var_deta:
    var_deta = list(ds_deta.data_vars)
deta_xa  = ds_deta[var_deta[0]]
time_dim = [d for d in deta_xa.dims if "time" in d.lower()][0]
t_deta   = pd.DatetimeIndex(deta_xa[time_dim].values)
deta = pd.Series(deta_xa.values,
                 index=pd.PeriodIndex(t_deta, freq="M"), name="Δη_NB-SB")

common    = transport.index.intersection(deta.index)
T_raw     = transport[common]
dE_raw    = deta[common]
t_dt      = common.to_timestamp()
print(f"  Période : {common[0]} → {common[-1]}  ({len(common)} mois)")

# ─────────────────────────────────────────────────────────────────
# 2. DÉSAISONNALISATION + LISSAGE 12 MOIS
# ─────────────────────────────────────────────────────────────────

def desaison(s):
    clim = s.groupby(s.index.month).transform("mean")
    return s - clim

def smooth_roll(s, w=SMOOTH):
    return s.rolling(window=w, center=True, min_periods=w // 2).mean()

T_sm  = smooth_roll(desaison(T_raw))
dE_sm = smooth_roll(desaison(dE_raw))

valid = T_sm.notna() & dE_sm.notna()
T_v   = T_sm[valid].values
dE_v  = dE_sm[valid].values
t_v   = t_dt[valid.values]
N     = len(T_v)
print(f"  Points valides après lissage : {N}")

# ─────────────────────────────────────────────────────────────────
# 3. STATISTIQUES
# ─────────────────────────────────────────────────────────────────

r, p_val = stats.pearsonr(dE_v, T_v)
slope, intercept, rv, _, se = stats.linregress(dE_v, T_v)
r2 = rv ** 2

# Intervalle de confiance 95% sur la pente (t-test bilatéral)
t_crit  = stats.t.ppf(0.975, df=N - 2)
se_pente = se
ci_slope = t_crit * se_pente

# Tendances linéaires individuelles (dérive dans chaque série)
t_num = np.arange(N)
sl_T,  ic_T,  _, p_T,  _ = stats.linregress(t_num, T_v)
sl_dE, ic_dE, _, p_dE, _ = stats.linregress(t_num, dE_v)

print(f"\n══════════════════════════════════════════════════════")
print(f"  r(T, Δη_NB−SB) = {r:+.4f}   p = {p_val:.2e}")
print(f"  R²             = {r2:.4f}   ({r2*100:.1f}% de variance expliquée)")
print(f"  Calibration    : T = {slope:.2f} (±{ci_slope:.2f}) · Δη  +  {intercept:.3f}  [Sv/m]")
print(f"  Tendance T     : {sl_T*12:.3f} Sv/an  (p={p_T:.3f})")
print(f"  Tendance Δη    : {sl_dE*12*1000:.3f} mm/an  (p={p_dE:.3f})")
print(f"══════════════════════════════════════════════════════")

# ─────────────────────────────────────────────────────────────────
# SAUVEGARDE TEXTE
# ─────────────────────────────────────────────────────────────────

out_txt = DIR_OUT / "corr_SBNB_results.txt"
with open(out_txt, "w") as f:
    f.write("r(T_Drake, Δη_NB-SB NEMO) — anomalies lissées 12 mois\n")
    f.write(f"Période : {common[0]} → {common[-1]}  ({N} points valides)\n\n")
    f.write(f"r     = {r:+.4f}\n")
    f.write(f"p     = {p_val:.2e}\n")
    f.write(f"R²    = {r2:.4f}  ({r2*100:.1f}%)\n")
    f.write(f"Pente = {slope:.3f} ± {ci_slope:.3f}  Sv/m  (IC 95%)\n")
    f.write(f"Intercept = {intercept:.4f}  Sv\n")
    f.write(f"Tendance T  : {sl_T*12:.4f} Sv/an  (p={p_T:.4f})\n")
    f.write(f"Tendance Δη : {sl_dE*12*1000:.4f} mm/an  (p={p_dE:.4f})\n")
print(f"✓ Résultats : {out_txt}")

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — Séries temporelles superposées (double axe Y)
# ─────────────────────────────────────────────────────────────────

fig1, ax_a = plt.subplots(figsize=(13, 5))
ax_b = ax_a.twinx()

ax_a.plot(t_v, T_v,  color="steelblue",  lw=2.2, label="T Drake  (Sv)")
ax_b.plot(t_v, dE_v, color="darkorange", lw=2.2, alpha=0.9, label="Δη NB−SB  (m)")
ax_a.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
ax_b.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

ax_a.set_ylabel("T'  (Sv)",  color="steelblue",  fontsize=15, fontweight="bold")
ax_b.set_ylabel("Δη'  (m)", color="darkorange", fontsize=15, fontweight="bold")
ax_a.tick_params(axis="y", labelcolor="steelblue",  labelsize=14)
ax_b.tick_params(axis="y", labelcolor="darkorange", labelsize=14)

ax_a.set_title(
    f"Anomalies de transport et de gradient SSH NB−SB — lissage {SMOOTH} mois\n"
    f"r = {r:+.3f}   R² = {r2:.2f}   p = {p_val:.2e}",
    fontsize=12, fontweight="bold")

lines = [plt.Line2D([0],[0], color="steelblue",  lw=2.2, label="T' Drake lissé (Sv)"),
         plt.Line2D([0],[0], color="darkorange", lw=2.2, label="Δη' NB−SB lissé (m)")]
ax_a.legend(handles=lines, fontsize=11, loc="upper left")
ax_a.grid(alpha=0.3)
ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_a.xaxis.set_major_locator(mdates.YearLocator(5))
plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
fig1.savefig(DIR_OUT / "corr_SBNB_timeseries.png", dpi=150, bbox_inches="tight")
print(f"✓ Figure 1 : {DIR_OUT}/corr_SBNB_timeseries.png")
plt.show()

# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — Nuage de points + droite de régression
# ─────────────────────────────────────────────────────────────────

fig2, ax = plt.subplots(figsize=(7, 6))

ax.scatter(dE_v, T_v, s=12, alpha=0.45, color="steelblue", zorder=2,
           label=f"n = {N} mois")

xfit = np.linspace(dE_v.min(), dE_v.max(), 300)
ax.plot(xfit, slope * xfit + intercept,
        color="firebrick", lw=2.2, zorder=3,
        label=f"T = {slope:.1f}·Δη + {intercept:.2f}\n"
              f"r = {r:+.3f}   R² = {r2:.2f}")

# Bande de confiance 95% sur la régression
x_mean = dE_v.mean()
se_fit = se_pente * np.sqrt(1/N + (xfit - x_mean)**2 / np.sum((dE_v - x_mean)**2))
y_fit  = slope * xfit + intercept
ax.fill_between(xfit,
                y_fit - t_crit * se_fit,
                y_fit + t_crit * se_fit,
                color="firebrick", alpha=0.12, label="IC 95% régression")

ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)

ax.set_xlabel("Δη' NB−SB lissé  (m)", fontsize=12)
ax.set_ylabel("T' Drake lissé  (Sv)",  fontsize=12)
ax.set_title(
    f"r(T_Drake, Δη_NB−SB) dans NEMO\n"
    f"Anomalies lissées {SMOOTH} mois  —  1991–2023",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig2.savefig(DIR_OUT / "corr_SBNB_scatter.png", dpi=150, bbox_inches="tight")
print(f"✓ Figure 2 : {DIR_OUT}/corr_SBNB_scatter.png")
plt.show()

print(f"\n✓ Terminé. Sorties dans {DIR_OUT}/")