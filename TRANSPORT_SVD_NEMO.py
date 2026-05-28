"""
Calcul du transport au passage de Drake — 1991-2023 (396 mois)
Source : transport_1991_2023.nc  (variable uocetr)

Transport [Sv] = (1/1e6) * Σ_{j,k} uocetr(t,k,j,i) * umask(k,j,i)

Sorties :
  - drake_transport_1991_2023.nc      : série temporelle en Sv
  - drake_transport_1991_2023.png     : série temporelle + lissage + tendance
  - drake_transect_map_1991_2023.png  : carte de localisation du transect
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES — même transect que le script annuel
# ─────────────────────────────────────────────────────────────────

LON_DRAKE = -68.0
LAT_SUD   = -67.0
LAT_NORD  = -55.0
SMOOTH    = 12   # lissage en mois

FILE_U    = "transport_1991_2023.nc"
FILE_MASK = "mask_continent.nc"

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────

print("Chargement des fichiers…")
ds_u    = xr.open_dataset(FILE_U,    chunks={"time_counter": 12})
ds_mask = xr.open_dataset(FILE_MASK)

print("  uocetr :", dict(ds_u.sizes))

# ─────────────────────────────────────────────────────────────────
# 2. MASQUE CONTINENT
# ─────────────────────────────────────────────────────────────────

umask = ds_mask["umask"]
if "time_counter" in umask.dims:
    umask = umask.isel(time_counter=0)
depth_dim = [d for d in umask.dims if d not in ("y", "x")][0]
if depth_dim != "depthu":
    umask = umask.rename({depth_dim: "depthu"})
assert umask.dims == ("depthu", "y", "x"), f"Dims inattendues : {umask.dims}"

# ─────────────────────────────────────────────────────────────────
# 3. COORDONNÉES  (nav_lon / nav_lat depuis le masque)
# ─────────────────────────────────────────────────────────────────

nav_lon = ds_mask["nav_lon"].values   # (y, x)
nav_lat = ds_mask["nav_lat"].values   # (y, x)

# ─────────────────────────────────────────────────────────────────
# 4. SÉLECTION DU TRANSECT DE DRAKE
# ─────────────────────────────────────────────────────────────────

j_ref   = int(np.argmin(np.abs(nav_lat[:, 0] - (-60.0))))
i_drake = int(np.argmin(np.abs(nav_lon[j_ref, :] - LON_DRAKE)))

umask_surf = umask.isel(depthu=0, x=i_drake).values
lat_col    = nav_lat[:, i_drake]
lon_real   = nav_lon[j_ref, i_drake]

mask_zone = (lat_col >= LAT_SUD) & (lat_col <= LAT_NORD) & (umask_surf == 1)
j_indices = np.where(mask_zone)[0]

lat_transect_sud  = lat_col[j_indices].min()
lat_transect_nord = lat_col[j_indices].max()

gaps = np.where(np.diff(j_indices) > 1)[0]

print(f"\nTransect : i={i_drake}, lon={lon_real:.2f}°")
print(f"  Latitudes : {lat_transect_sud:.2f}° → {lat_transect_nord:.2f}°")
print(f"  Points j  : {len(j_indices)}")
if len(gaps) > 0:
    print(f"  ℹ {len(gaps)} île(s) masquée(s) dans la fenêtre")

# ─────────────────────────────────────────────────────────────────
# 5. CALCUL DU TRANSPORT (1991-2023)
# ─────────────────────────────────────────────────────────────────

uocetr       = ds_u["uocetr"]                                        # (t, depthu, y, x)
uocetr_drake = uocetr.isel(x=i_drake, y=j_indices.tolist())          # (t, depthu, j_sel)
umask_drake  = umask.isel( x=i_drake, y=j_indices.tolist())          # (depthu, j_sel)

print("\nCalcul du transport (peut prendre quelques secondes)…")
transport_sv = (uocetr_drake * umask_drake).sum(dim=["depthu", "y"]) / 1e6
transport_sv = transport_sv.compute()

transport_sv.name = "drake_transport"
transport_sv.attrs = {
    "units"    : "Sv",
    "long_name": "Drake Passage transport (est = positif)",
    "lon"      : f"{lon_real:.2f}°",
    "lat_range": f"{lat_transect_sud:.1f}° à {lat_transect_nord:.1f}°",
}

t_pd = pd.DatetimeIndex(transport_sv["time_counter"].values)
vals = transport_sv.values

print(f"\n── Statistiques 1991-2023 ────────────────────────")
print(f"  Moy : {vals.mean():.1f} Sv")
print(f"  Min : {vals.min():.1f} Sv  ({str(t_pd[vals.argmin()])[:10]})")
print(f"  Max : {vals.max():.1f} Sv  ({str(t_pd[vals.argmax()])[:10]})")
print(f"  Std : {vals.std():.1f} Sv")

# ─────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE NetCDF
# ─────────────────────────────────────────────────────────────────

transport_sv.to_netcdf("drake_transport_1991_2023.nc")
print("\n✓ Sauvegardé : drake_transport_1991_2023.nc")

# ─────────────────────────────────────────────────────────────────
# 7. LISSAGE + TENDANCE
# ─────────────────────────────────────────────────────────────────

series  = pd.Series(vals, index=t_pd)
lissee  = series.rolling(window=SMOOTH, center=True,
                          min_periods=SMOOTH // 2).mean()

t_num = np.arange(len(vals))
slope, intercept, _, p_val, _ = linregress(t_num, vals)
trend = slope * t_num + intercept
sig   = "✓" if p_val < 0.05 else "(non sig.)"

print(f"  Tendance : {slope*12:.3f} Sv/an  (p={p_val:.3f})  {sig}")

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — Série temporelle 1991-2023
# ─────────────────────────────────────────────────────────────────

mean_val = vals.mean()

fig1, ax = plt.subplots(figsize=(13, 5))

ax.plot(t_pd, vals,
        color="steelblue", lw=0.9, alpha=0.35, label="Transport mensuel")
ax.plot(t_pd, lissee.values,
        color="steelblue", lw=2.2, label=f"Lissage {SMOOTH} mois")
ax.axhline(mean_val, color="darkorange", ls="--", lw=1.5, alpha=0.5,
           label=f"Moyenne = {mean_val:.1f} Sv")
ax.plot(t_pd, trend,
        color="firebrick", lw=1.2, ls="-.",
        label=f"Tendance : {slope*12:.3f} Sv/an  {sig}")

ax.set_ylabel("Transport (Sv)", fontsize=12)
ax.set_title("Transport au passage de Drake — 1991–2023 (mensuel)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
fig1.savefig("drake_transport_1991_2023.png", dpi=150, bbox_inches="tight")
print("✓ Figure transport : drake_transport_1991_2023.png")
plt.show()

# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — Carte de localisation du transect
# ─────────────────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(
    figsize=(7, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
ax2.set_extent([-90, -50, -72, -48], crs=ccrs.PlateCarree())

ax2.add_feature(cfeature.OCEAN,     color="#cde6f5", zorder=0)
ax2.add_feature(cfeature.LAND,      color="#c8b89a", zorder=1)
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555", zorder=2)
ax2.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#888888",
                linestyle=":", zorder=2)
ax2.add_feature(
    cfeature.NaturalEarthFeature(
        "physical", "land", "50m",
        facecolor="#c8b89a", edgecolor="#555555", linewidth=0.8
    ),
    zorder=1
)

ax2.plot(
    [lon_real, lon_real],
    [lat_transect_sud, lat_transect_nord],
    color="red", lw=2.5, transform=ccrs.PlateCarree(), zorder=5
)
ax2.text(
    lon_real + 1.0,
    (lat_transect_sud + lat_transect_nord) / 2,
    f"Transect\n({lon_real:.1f}°W)\n{lat_transect_sud:.1f}° → {lat_transect_nord:.1f}°",
    color="red", fontsize=8, fontweight="bold",
    va="center", ha="left",
    transform=ccrs.PlateCarree(), zorder=6,
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2)
)

gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                   alpha=0.6, linestyle="--")
gl.top_labels   = False
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

ax2.set_title("Localisation du transect — Passage de Drake (1991–2023)",
              fontsize=11, fontweight="bold", pad=10)

fig2.savefig("drake_transect_map_1991_2023.png", dpi=150, bbox_inches="tight")
print("✓ Figure carte    : drake_transect_map_1991_2023.png")
plt.show()
