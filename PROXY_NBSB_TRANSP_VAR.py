"""
Gradient dynamique de surface au passage de Drake  —  Δη = η_NB − η_SB
  Fronts définis par des contours SSH (Southern / Northern Boundary ACC)

  OBS  : AVISO/DUACS  (dot_all_30bmedian_eigen6s4v2_sig3.nc)  2002–2018
  NEMO : ZOS          (zos_1991_2023_IS.nc)                   1991–2023

  Valeurs SSH des fronts au passage de Drake :
      NB  →  OBS : −0.620 m  |  NEMO : −0.300 m
      SB  →  OBS : −2.080 m  |  NEMO : −1.600 m

  Méthode (identique à PROXY_TRANSP_VARIABILITY.py) :
    - On localise la latitude où la SSH moyenne temporelle = valeur de front
    - On moyenne N_AVG points océaniques autour de cette latitude
    - Δη = η_NB − η_SB

  Figure 1 — Δη brut + moyenne + bande ±1σ  (stabilité absolue)
  Figure 2 — Δη désaisonnalisé lissé + tendance  (dérive basse fréquence)

  Sorties dans ./outputs_SBNB_proxytransp/
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────

I_DRAKE    = 875
LON_DRAKE  = -68.12

# Valeurs SSH définissant les fronts au passage de Drake
NB_SSH_OBS  = -0.620   # m
SB_SSH_OBS  = -2.080   # m
NB_SSH_NEMO = -0.300   # m
SB_SSH_NEMO = -1.600   # m

N_AVG_NEMO = 4    # nb de points NEMO à moyenner autour du front
N_AVG_OBS  = 2    # nb de points OBS  à moyenner autour du front
SMOOTH     = 12   # lissage en mois (1 an)

DIR_MAIN      = Path(".")
DIR_TRANSPORT = Path("./Transport_sv")
DIR_OUT       = Path("./outputs_SBNB_proxytransp")
DIR_OUT.mkdir(parents=True, exist_ok=True)

FILE_OBS  = DIR_MAIN      / "dot_all_30bmedian_eigen6s4v2_sig3.nc"
FILE_MOD  = DIR_MAIN      / "zos_1991_2023_IS.nc"
FILE_MASK = DIR_TRANSPORT / "mask_continent.nc"

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────

ds_obs  = xr.open_dataset(FILE_OBS)
ds_mod  = xr.open_dataset(FILE_MOD)
ds_mask = xr.open_dataset(FILE_MASK)

nav_lat = ds_mask["nav_lat"].values
umask   = ds_mask["umask"]
if "time_counter" in umask.dims:
    umask = umask.isel(time_counter=0)
depth_dim = [d for d in umask.dims if d not in ("y", "x")][0]
if depth_dim != "depthu":
    umask = umask.rename({depth_dim: "depthu"})
umask_surf = umask.isel(depthu=0)

# ─────────────────────────────────────────────────────────────────
# 2. LOCALISATION DES FRONTS PAR CONTOUR SSH
#
#    Pour chaque source (OBS / NEMO) :
#      → on calcule la SSH moyenne temporelle sur la colonne Drake
#      → on trouve la latitude où SSH_mean ≈ valeur du front
#      → on retient les N_AVG points océaniques les plus proches
# ─────────────────────────────────────────────────────────────────

# ── NEMO ──────────────────────────────────────────────────────────
lat_col  = nav_lat[:, I_DRAKE]
mask_col = umask_surf.isel(x=I_DRAKE).values   # 1 = océan, 0 = terre

zos        = ds_mod["zos"]
zos_col    = zos.isel(x=I_DRAKE)               # (time, y)
ssh_mean_nemo = zos_col.mean(dim="time_counter").values  # moyenne temporelle

def find_j_front_nemo(lat_col, mask_col, ssh_mean, ssh_target, n):
    """Indices NEMO : n points océaniques où SSH_mean est la plus proche de ssh_target."""
    dist = np.abs(ssh_mean - ssh_target)
    dist[mask_col == 0] = np.inf          # exclure les points terrestres
    return np.sort(np.argsort(dist)[:n])

j_NB_nemo = find_j_front_nemo(lat_col, mask_col, ssh_mean_nemo, NB_SSH_NEMO, N_AVG_NEMO)
j_SB_nemo = find_j_front_nemo(lat_col, mask_col, ssh_mean_nemo, SB_SSH_NEMO, N_AVG_NEMO)

print("─── NEMO ─────────────────────────────────────────────────────")
print(f"  NB : SSH cible = {NB_SSH_NEMO} m  →  lat moy = {lat_col[j_NB_nemo].mean():.2f}°"
      f"   SSH moy = {ssh_mean_nemo[j_NB_nemo].mean():.3f} m")
print(f"  SB : SSH cible = {SB_SSH_NEMO} m  →  lat moy = {lat_col[j_SB_nemo].mean():.2f}°"
      f"   SSH moy = {ssh_mean_nemo[j_SB_nemo].mean():.3f} m")

# ── OBS ───────────────────────────────────────────────────────────
lats = ds_obs["latitude"].values
lons = ds_obs["longitude"].values
i_obs = int(np.argmin(np.abs(lons - LON_DRAKE)))
land  = ds_obs["land_mask"].isel(longitude=i_obs).values   # 1 = terre

dot          = ds_obs["dot"]
dot_col      = dot.isel(longitude=i_obs)                   # (time, latitude)
ssh_mean_obs = dot_col.mean(dim="time").values              # moyenne temporelle

def find_j_front_obs(lats, land, ssh_mean, ssh_target, n):
    """Indices OBS : n points océaniques où SSH_mean est la plus proche de ssh_target."""
    dist = np.abs(ssh_mean - ssh_target)
    dist[land == 1] = np.inf              # exclure les points terrestres
    return np.sort(np.argsort(dist)[:n])

j_NB_obs = find_j_front_obs(lats, land, ssh_mean_obs, NB_SSH_OBS, N_AVG_OBS)
j_SB_obs = find_j_front_obs(lats, land, ssh_mean_obs, SB_SSH_OBS, N_AVG_OBS)

print("─── OBS ──────────────────────────────────────────────────────")
print(f"  NB : SSH cible = {NB_SSH_OBS} m  →  lat moy = {lats[j_NB_obs].mean():.2f}°"
      f"   SSH moy = {ssh_mean_obs[j_NB_obs].mean():.3f} m")
print(f"  SB : SSH cible = {SB_SSH_OBS} m  →  lat moy = {lats[j_SB_obs].mean():.2f}°"
      f"   SSH moy = {ssh_mean_obs[j_SB_obs].mean():.3f} m")

# ─────────────────────────────────────────────────────────────────
# 3. CALCUL DE Δη = η_NB − η_SB
# ─────────────────────────────────────────────────────────────────

# ── NEMO ──────────────────────────────────────────────────────────
eta_SB_mod   = zos_col.isel(y=j_SB_nemo.tolist()).mean(dim="y")
eta_NB_mod   = zos_col.isel(y=j_NB_nemo.tolist()).mean(dim="y")
delta_mod_xa = eta_NB_mod - eta_SB_mod
t_mod_pd     = pd.DatetimeIndex(delta_mod_xa["time_counter"].values)

# ── OBS ───────────────────────────────────────────────────────────
eta_SB_obs   = dot_col.isel(latitude=j_SB_obs.tolist()).mean(dim="latitude")
eta_NB_obs   = dot_col.isel(latitude=j_NB_obs.tolist()).mean(dim="latitude")
delta_obs_xa = eta_NB_obs - eta_SB_obs
t_obs_pd     = pd.DatetimeIndex(delta_obs_xa["time"].values)

# ─────────────────────────────────────────────────────────────────
# 4. SÉRIES PANDAS + ALIGNEMENT TEMPOREL
# ─────────────────────────────────────────────────────────────────

mod_full = pd.Series(delta_mod_xa.values,
                     index=pd.PeriodIndex(t_mod_pd, freq="M"))
obs_full = pd.Series(delta_obs_xa.values,
                     index=pd.PeriodIndex(t_obs_pd, freq="M"))

common_idx  = mod_full.index.intersection(obs_full.index)
mod_common  = mod_full[common_idx]
obs_common  = obs_full[common_idx]
t_common_dt = common_idx.to_timestamp()

print(f"\nPériode commune : {common_idx[0]} → {common_idx[-1]}  ({len(common_idx)} mois)")

# ─────────────────────────────────────────────────────────────────
# 5. DÉSAISONNALISATION + LISSAGE
# ─────────────────────────────────────────────────────────────────

def desaison_smooth(series, window):
    clim   = series.groupby(series.index.month).transform("mean")
    anom   = series - clim
    smooth = anom.rolling(window=window, center=True,
                          min_periods=window // 2).mean()
    return anom, smooth

mod_anom, mod_smooth = desaison_smooth(mod_common, SMOOTH)
obs_anom, obs_smooth = desaison_smooth(obs_common, SMOOTH)

_, mod_full_smooth = desaison_smooth(mod_full, SMOOTH)

valid      = mod_smooth.notna() & obs_smooth.notna()
t_valid_dt = t_common_dt[valid.values]

# ─────────────────────────────────────────────────────────────────
# 6. STATISTIQUES
# ─────────────────────────────────────────────────────────────────

moy_mod = mod_common.mean()
moy_obs = obs_common.mean()
std_mod = mod_common.std()
std_obs = obs_common.std()
cv_mod  = std_mod / moy_mod * 100
cv_obs  = std_obs / moy_obs * 100
biais   = moy_mod - moy_obs

r_sm, p_sm = stats.pearsonr(mod_smooth[valid].values,
                             obs_smooth[valid].values)

t_num = np.arange(valid.sum())
sl_mod, ic_mod, _, p_tr_mod, _ = stats.linregress(t_num, mod_smooth[valid].values)
sl_obs, ic_obs, _, p_tr_obs, _ = stats.linregress(t_num, obs_smooth[valid].values)
trend_mod = sl_mod * t_num + ic_mod
trend_obs = sl_obs * t_num + ic_obs

print(f"\n── Stabilité de Δη (NB−SB) ───────────────────────────────────")
print(f"  Moy Δη NEMO : {moy_mod:.3f} m   σ = {std_mod:.3f} m   CV = {cv_mod:.1f}%")
print(f"  Moy Δη OBS  : {moy_obs:.3f} m   σ = {std_obs:.3f} m   CV = {cv_obs:.1f}%")
print(f"  Biais NEMO−OBS : {biais:.3f} m")
print(f"  r lissé : {r_sm:.3f}  (p = {p_sm:.3f})  {'✓' if p_sm < 0.05 else '✗'}")
print(f"  Tendance NEMO : {sl_mod*12*1000:.2f} mm/an  (p={p_tr_mod:.3f})")
print(f"  Tendance OBS  : {sl_obs*12*1000:.2f} mm/an  (p={p_tr_obs:.3f})")

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — STABILITÉ ABSOLUE de Δη (valeurs brutes)
# ─────────────────────────────────────────────────────────────────

fig1, ax = plt.subplots(figsize=(13, 4.5))

ax.plot(t_common_dt, mod_common.values,
        color="steelblue", lw=1.0, alpha=0.35)
ax.plot(t_common_dt, obs_common.values,
        color="darkorange", lw=1.0, alpha=0.35)

ax.axhline(moy_mod, color="steelblue", lw=2.0,
           label=f"NEMO  moy = {moy_mod:.3f} m   σ = {std_mod:.3f} m   CV = {cv_mod:.1f}%")
ax.axhline(moy_obs, color="darkorange", lw=2.0,
           label=f"OBS   moy = {moy_obs:.3f} m   σ = {std_obs:.3f} m   CV = {cv_obs:.1f}%")

ax.fill_between(t_common_dt,
                moy_mod - std_mod, moy_mod + std_mod,
                color="steelblue", alpha=0.12, label="_")
ax.fill_between(t_common_dt,
                moy_obs - std_obs, moy_obs + std_obs,
                color="darkorange", alpha=0.15, label="_")

ax.set_ylabel("Δη  (m)", fontsize=12)
ax.set_title(
    f"Stabilité du gradient SSH Δη = η_NB − η_SB au passage de Drake\n"
    f"Période commune 2002–2018  —  "
    f"NB : {NB_SSH_OBS} m (obs) / {NB_SSH_NEMO} m (NEMO)  |  "
    f"SB : {SB_SSH_OBS} m (obs) / {SB_SSH_NEMO} m (NEMO)",
    fontsize=10, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig1.savefig(DIR_OUT / "delta_eta_SBNB_stabilite.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Figure 1 : {DIR_OUT}/delta_eta_SBNB_stabilite.png")
plt.show()

# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — VARIABILITÉ BASSE FRÉQUENCE + TENDANCE
# ─────────────────────────────────────────────────────────────────

fig2, axes = plt.subplots(2, 1, figsize=(13, 8))

# ── Panel 1 : vue globale ──────────────────────────────────────
ax1 = axes[0]
ax1.plot(mod_full.index.to_timestamp(), mod_full_smooth.values,
         color="steelblue", lw=1.8, label="NEMO (1991–2023)")
ax1.plot(t_obs_pd, desaison_smooth(obs_full, SMOOTH)[1].values,
         color="darkorange", lw=1.8, label="OBS AVISO (2002–2018)")
ax1.axhline(0, color="k", lw=0.5, ls="--")
ax1.set_ylabel("Δη'  (m)", fontsize=12)
ax1.set_title(
    f"Anomalie de Δη (NB−SB) désaisonnalisée — lissage {SMOOTH} mois",
    fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(5))

# ── Panel 2 : période commune + tendances ─────────────────────
ax2 = axes[1]

ax2.plot(t_common_dt, mod_anom.values,
         color="steelblue", lw=0.5, alpha=0.2)
ax2.plot(t_common_dt, obs_anom.values,
         color="darkorange", lw=0.5, alpha=0.2)

ax2.plot(t_common_dt, mod_smooth.values,
         color="steelblue", lw=2.2,
         label=f"NEMO  lissé {SMOOTH} mois")
ax2.plot(t_common_dt, obs_smooth.values,
         color="darkorange", lw=2.2,
         label=f"OBS   lissé {SMOOTH} mois")

sig_mod = "✓" if p_tr_mod < 0.05 else "(non sig.)"
sig_obs = "✓" if p_tr_obs < 0.05 else "(non sig.)"
ax2.plot(t_valid_dt, trend_mod,
         color="steelblue", lw=1.5, ls="--",
         label=f"Tendance NEMO : {sl_mod*12*1000:.2f} mm/an  {sig_mod}")
ax2.plot(t_valid_dt, trend_obs,
         color="darkorange", lw=1.5, ls="--",
         label=f"Tendance OBS  : {sl_obs*12*1000:.2f} mm/an  {sig_obs}")

ax2.axhline(0, color="k", lw=0.5, ls=":")
ax2.set_ylabel("Δη'  (m)", fontsize=12)
ax2.set_title(
    f"Période commune 2002–2018  —  r = {r_sm:.3f}  (p = {p_sm:.3f})",
    fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig2.savefig(DIR_OUT / "delta_eta_SBNB_tendance.png", dpi=150, bbox_inches="tight")
print(f"✓ Figure 2 : {DIR_OUT}/delta_eta_SBNB_tendance.png")
plt.show()

# ─────────────────────────────────────────────────────────────────
# SAUVEGARDE NetCDF
# ─────────────────────────────────────────────────────────────────

delta_mod_xa.to_netcdf(DIR_OUT / "delta_eta_SBNB_NEMO_1991_2023.nc")
delta_obs_xa.to_netcdf(DIR_OUT / "delta_eta_SBNB_OBS_2002_2018.nc")
print(f"✓ {DIR_OUT}/delta_eta_SBNB_NEMO_1991_2023.nc")
print(f"✓ {DIR_OUT}/delta_eta_SBNB_OBS_2002_2018.nc")