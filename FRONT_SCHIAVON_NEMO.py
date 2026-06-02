"""
Identification des fronts de l'ACC — Sokolov & Rintoul (2009a)
===============================================================
Données  : ZOS mensuel NEMO 1/4°, 1991–2023  (zos_1991_2023_IS.nc)
Masque   : ./Transport_sv/mask_continent.nc  (non-zéro = océan)
Output   : ./outputs_park_v2/schiavon_NEMO_fronts_SR2009a_<N>sec.png

Méthode (S&R 2009a, Section 2) :
  1. Calcul de |∇η| = sqrt((∂η/∂x)² + (∂η/∂y)²)  en m/100 km
  2. Points actifs : |∇η| > 0.25 m/100 km  OU  maximum local de |∇η|
  3. Par secteur de LON_STEP° :
       · Histogramme SSH pondéré par |∇η| sur les points actifs
       · 3 pics → valeurs SSH optimales ζᵢ (SAF, PF, SACCF) par secteur
  4. Label circumpolaire = médiane des ζᵢ par secteur
  5. Chemin du front = CONTOUR SSH = ζᵢ sur le champ ZOS moyen
     → c'est exactement ce que font S&R (leurs Fig. 6 & 7)
     → le chemin suit naturellement la topographie et les méandres,
       pas une ligne droite entre des points de secteur

Nota :
  - Le masque (non-zéro=océan) est appliqué en premier sur ZOS.
    Tous les calculs ultérieurs utilisent isfinite(zos_mean) comme filtre.
  - Les étapes 3–4 restent les mêmes ; seul l'affichage change.
"""

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os

# =============================================================================
# PARAMÈTRES
# =============================================================================

FILEPATH   = "zos_1991_2023_IS.nc"
MASK_FILE  = "./Transport_sv/mask_continent.nc"
OUTPUT_DIR = "./outputs_park_v2/"
OUTPUT_PFX = "schiavon_NEMO_"

LAT_MIN = -70.0   # °S
LAT_MAX = -30.0   # °S

# ─── NOMBRE DE SECTEURS — CHANGER ICI POUR TESTER ────────────────────────────
#   12  →  30°  (S&R 2009a original)
#   18  →  20°
#   36  →  10°  (Schiavon 2025)
#   72  →   5°
N_SECTORS = 36    # ← MODIFIER ICI
# ─────────────────────────────────────────────────────────────────────────────

LON_STEP       = 360.0 / N_SECTORS
GRAD_THRESHOLD = 0.25       # m/100 km  (S&R 2009a)
R_EARTH        = 6_371_000.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CHARGEMENT + MASQUE CONTINENT
# =============================================================================

print("=" * 60)
print(f"  S&R 2009a — {N_SECTORS} secteurs × {LON_STEP:.0f}°")
print("=" * 60)

print("\n[1/5] Chargement SSH + masque continent...")

ds      = xr.open_dataset(FILEPATH)
zos     = ds["zos"]
nav_lat = ds["nav_lat"].values
nav_lon = ds["nav_lon"].values
nav_lon = np.where(nav_lon < 0, nav_lon + 360, nav_lon)

# Masque continent : non-zéro = océan (confirmé par l'utilisateur)
ds_mask   = xr.open_dataset(MASK_FILE)
umask     = ds_mask["umask"]
if "time_counter" in umask.dims:
    umask = umask.isel(time_counter=0)
depth_dim = [d for d in umask.dims if d not in ("y", "x")][0]
umask_2d  = umask.isel({depth_dim: 0}).values

ocean_mask = (umask_2d != 0)   # True = océan

if ocean_mask.shape != nav_lat.shape:
    print("   ⚠ Shape masque incompatible — masque ignoré")
    ocean_mask = np.ones(nav_lat.shape, dtype=bool)
else:
    print(f"   Masque OK — {ocean_mask.sum():,} points océan "
          f"({100*ocean_mask.mean():.1f}% de la grille)")

# =============================================================================
# 2. MOYENNE TEMPORELLE + APPLICATION DU MASQUE
# =============================================================================

print("\n[2/5] Moyenne temporelle 1991–2023...")

zos_mean = zos.mean(dim="time_counter").values

# ── Appliquer le masque continent UNE FOIS ici ──────────────────────────────
# Tous les calculs suivants utilisent np.isfinite(zos_mean) comme filtre océan
zos_mean = np.where(ocean_mask, zos_mean, np.nan)

in_band = (nav_lat >= LAT_MIN) & (nav_lat <= LAT_MAX) & np.isfinite(zos_mean)
print(f"   Points océan valides dans {LAT_MIN}°–{LAT_MAX}° : {in_band.sum():,}")

if in_band.sum() < 1000:
    raise RuntimeError(
        f"Seulement {in_band.sum()} points valides — vérifier le masque."
    )

# =============================================================================
# 3. GRADIENT SSH 2D   |∇η|   (m/100 km)
# =============================================================================

print("\n[3/5] Calcul de |∇η| 2D sur grille curvilinéaire...")

lat_rad = np.deg2rad(nav_lat)
lon_rad = np.deg2rad(nav_lon)

dy = np.gradient(lat_rad, axis=0) * R_EARTH
dx = np.gradient(lon_rad, axis=1) * R_EARTH * np.cos(lat_rad)

with np.errstate(invalid="ignore", divide="ignore"):
    deta_dy = np.gradient(zos_mean, axis=0) / dy
    deta_dx = np.gradient(zos_mean, axis=1) / dx

grad_m100km = np.sqrt(deta_dx**2 + deta_dy**2) * 1e5
grad_m100km = np.where(in_band, grad_m100km, np.nan)

print(f"   |∇η| max : {np.nanmax(grad_m100km):.3f} m/100 km")
n_above = np.nansum(grad_m100km[in_band] > GRAD_THRESHOLD)
print(f"   Points > {GRAD_THRESHOLD} m/100 km : {n_above:,} "
      f"({100*n_above/in_band.sum():.1f}% de la bande)")

# =============================================================================
# 4. OPTIMISATION PAR SECTEUR — S&R 2009a
#    → trouve la valeur SSH ζᵢ optimale pour chaque front dans chaque secteur
# =============================================================================

print(f"\n[4/5] Optimisation des contours SSH dans {N_SECTORS} secteurs...")

local_max = (grad_m100km == maximum_filter(
    np.where(np.isfinite(grad_m100km), grad_m100km, 0.0), size=5))

active = ((grad_m100km > GRAD_THRESHOLD) | local_max) & in_band

lon_centers = np.arange(LON_STEP / 2, 360, LON_STEP)

# Stockage des ζᵢ par front et par secteur
ssh_labels = {f: [] for f in ["SAF", "PF", "SACCF"]}

LAT_RES_PROFILE = 0.5   # ° — résolution du profil méridional

for lon_c in lon_centers:
    lon_w = lon_c - LON_STEP / 2
    lon_e = lon_c + LON_STEP / 2

    if lon_w < 0:
        in_sec = (nav_lon >= lon_w + 360) | (nav_lon < lon_e)
    elif lon_e > 360:
        in_sec = (nav_lon >= lon_w) | (nav_lon < lon_e - 360)
    else:
        in_sec = (nav_lon >= lon_w) & (nav_lon < lon_e)

    pts_act = active & in_sec

    if pts_act.sum() < 20:
        for f in ssh_labels:
            ssh_labels[f].append(np.nan)
        continue

    ssh_pts  = zos_mean[pts_act]
    grad_pts = np.where(np.isfinite(grad_m100km[pts_act]),
                        grad_m100km[pts_act], 0.0)

    lo = np.percentile(ssh_pts, 3)
    hi = np.percentile(ssh_pts, 97)
    if hi <= lo:
        for f in ssh_labels:
            ssh_labels[f].append(np.nan)
        continue

    bins     = np.linspace(lo, hi, 301)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2.0
    mask_rng = (ssh_pts >= lo) & (ssh_pts <= hi)
    hist, _  = np.histogram(ssh_pts[mask_rng], bins=bins,
                             weights=grad_pts[mask_rng])
    hist_sm  = uniform_filter1d(hist.astype(float), size=9)

    if hist_sm.max() == 0:
        for f in ssh_labels:
            ssh_labels[f].append(np.nan)
        continue

    peaks, props = find_peaks(hist_sm,
                               prominence=hist_sm.max() * 0.03,
                               distance=6)

    if len(peaks) == 0:
        for f in ssh_labels:
            ssh_labels[f].append(np.nan)
        continue

    # 3 pics les plus proéminents, triés SSH décroissant (N → S)
    peak_ssh  = bin_ctrs[peaks]
    peak_prom = props["prominences"]
    best_idx  = np.argsort(peak_prom)[::-1][:3]
    top_ssh   = np.sort(peak_ssh[best_idx])[::-1]
    while len(top_ssh) < 3:
        top_ssh = np.append(top_ssh, np.nan)

    for f, zeta in zip(["SAF", "PF", "SACCF"], top_ssh):
        ssh_labels[f].append(float(zeta))

# ── Label circumpolaire = médiane des valeurs par secteur ────────────────────
zeta_circ = {}
print("\n   Labels SSH circumpolaires :")
for f in ["SAF", "PF", "SACCF"]:
    vals = [v for v in ssh_labels[f] if not np.isnan(v)]
    if len(vals) == 0:
        zeta_circ[f] = np.nan
        print(f"   {f:6s} : aucun secteur valide !")
    else:
        zeta_circ[f] = float(np.median(vals))
        print(f"   {f:6s} : ζ = {zeta_circ[f]:.4f} m  "
              f"(σ = {np.std(vals):.4f} m,  {len(vals)}/{N_SECTORS} secteurs)")

# =============================================================================
# 5. INTERPOLATION SUR GRILLE RÉGULIÈRE
#    → ZOS et |∇η| pour l'affichage
# =============================================================================

print("\n[5/5] Interpolation sur grille régulière et carte...")

res      = 0.25
lon_reg  = np.arange(0, 360, res)
lat_reg  = np.arange(LAT_MIN, LAT_MAX + res, res)
LON_G, LAT_G = np.meshgrid(lon_reg, lat_reg)

ok = in_band & np.isfinite(grad_m100km)
pts_xy = np.column_stack([nav_lon[ok], nav_lat[ok]])

grad_reg = griddata(pts_xy, grad_m100km[ok], (LON_G, LAT_G), method="linear")
zos_reg  = griddata(pts_xy, zos_mean[ok],    (LON_G, LAT_G), method="linear")

# =============================================================================
# 6. CARTE  —  fond |∇η| + chemins des fronts comme contours SSH
#
# Méthode S&R : le chemin du front EST le contour SSH = ζᵢ
# sur le champ moyen. Pas de lignes droites entre points de secteur.
# =============================================================================

STYLES = {
    "SAF":   {"color": "#1f77b4", "label": "SAF",   "lw": 2.5},
    "PF":    {"color": "#2ca02c", "label": "PF",    "lw": 2.5},
    "SACCF": {"color": "#d62728", "label": "SACCF", "lw": 2.5},
}

fig, ax = plt.subplots(figsize=(17, 6))
ax.set_facecolor("#aaaaaa")

# Fond : |∇η|
vmax = np.nanpercentile(grad_reg, 97)
pcm  = ax.pcolormesh(lon_reg, lat_reg, grad_reg,
                     cmap="YlOrRd", vmin=0, vmax=vmax,
                     shading="auto", rasterized=True)
cbar = plt.colorbar(pcm, ax=ax,
                    orientation="vertical",
                    location="right",
                    fraction=0.025, pad=0.02, shrink=0.8,
                    label=r"$|\nabla\eta|$ (m 100 km$^{-1}$)")
# ── Fronts : contours du champ ZOS moyen aux valeurs ζᵢ ─────────────────────
# C'est exactement la méthode S&R : le chemin suit les streamlines SSH,
# ce qui donne des courbes continues qui épousent topographie et méandres.
for fname, style in STYLES.items():
    zeta = zeta_circ[fname]
    if np.isnan(zeta):
        continue

    # matplotlib.contour trace le contour SSH = ζ sur la grille régulière
    cs = ax.contour(lon_reg, lat_reg, zos_reg,
                    levels=[zeta],
                    colors=[style["color"]],
                    linewidths=style["lw"],
                    zorder=7)

    # Proxy pour la légende (contour n'a pas de handle direct)
    ax.plot([], [], color=style["color"], lw=style["lw"],
            label=f"{style['label']}  (ζ = {zeta:.3f} m)")

# Délimitation optionnelle des secteurs
for lon_edge in np.arange(0, 360, LON_STEP):
    ax.axvline(lon_edge, color="white", lw=0.4, alpha=0.3, linestyle=":")

ax.set_xlim(0, 360)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Latitude (°)", fontsize=12)
ax.set_title(
    "Fronts circumpolaires de l'ACC — Sokolov & Rintoul (2009a)\n"
    f"NEMO EXP_95_IS · Moyenne 1991–2023 · "
    f"{N_SECTORS} secteurs × {LON_STEP:.0f}°  "
    r"— chemins = contours $\eta$ = ζᵢ",
    fontsize=12
)
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.12, color="white", linestyle="--")

for lon_m, lab in [(10, "Atlantique"), (83, "Indien"), (250, "Pacifique")]:
    ax.text(lon_m, LAT_MAX - 1, lab, fontsize=9, color="white",
            ha="center", va="top", fontweight="bold")

plt.tight_layout()
outfile = os.path.join(OUTPUT_DIR,
          f"{OUTPUT_PFX}fronts_SR2009a_{N_SECTORS}sec.png")
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"\n✓ Sauvegardé : {outfile}")
plt.show()