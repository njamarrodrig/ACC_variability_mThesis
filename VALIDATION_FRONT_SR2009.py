"""
Fronts ACC (S&R 2009a, NEMO) — deux cartes
============================================
Output 1 : carte polaire stéréo avec |∇η| moyen en fond + fronts S&R calculés
           → schiavon_NEMO_SR2009a_polar_grad.png

Output 2 : carte polaire stéréo fronts S&R calculés vs littérature (GeoJSON)
           Légende :
             • "Type de front" : couleur par type (lit.) + noir linestyle (calculé)
             • "Source"        : linestyle par source
           → schiavon_NEMO_SR2009a_vs_lit.png
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.interpolate import griddata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape, LineString, MultiLineString

# =============================================================================
# PARAMÈTRES
# =============================================================================

FILEPATH    = "zos_1991_2023_IS.nc"
MASK_FILE   = "./Transport_sv/mask_continent.nc"
GEOJSON     = "/cyfast/njamar/method_fronts/fronts_positions.geojson"
OUTPUT_DIR  = "./outputs_park_v2/"

LAT_MIN = -80.0
LAT_MAX = -35.0

# ─── NOMBRE DE SECTEURS ──────────────────────────────────────────────────────
#   12  →  30°  (S&R 2009a original)
#   18  →  20°
#   36  →  10°  (Schiavon 2025)
N_SECTORS = 36    # ← MODIFIER ICI
# ─────────────────────────────────────────────────────────────────────────────

LON_STEP       = 360.0 / N_SECTORS
GRAD_THRESHOLD = 0.25
R_EARTH        = 6_371_000.0

# Contrainte de latitude pour le PF :
# Le SSH décroissant vers le sud, on calcule le SSH moyen à PF_LAT_MAX
# dans chaque secteur. Si le pic PF a un SSH supérieur à cette valeur,
# il correspond à une latitude trop au nord → on le remplace par le
# premier pic valide sous ce seuil SSH.
PF_LAT_MAX = -45.0   # ← MODIFIER ICI si besoin

# Fronts calculés : NOIRS, styles de ligne distincts
CALC_STYLES = {
    "SAF":   {"color": "black", "lw": 2.2, "ls": "-"},
    "PF":    {"color": "black", "lw": 2.2, "ls": "--"},
    "SACCF": {"color": "black", "lw": 2.2, "ls": ":"},
}

# Fronts littérature : couleur par type, semi-transparents
LIT_COLORS     = {"SAF": "crimson", "PF": "royalblue", "SACCF": "forestgreen"}
LIT_ALPHA      = 0.45
LIT_LW         = 1.2
LIT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# ÉTAPE 1 — CHARGEMENT + MASQUE
# =============================================================================

print("=" * 60)
print(f"  S&R 2009a — {N_SECTORS} × {LON_STEP:.0f}°")
print("=" * 60)
print("\n[1/5] Chargement SSH + masque...")

ds      = xr.open_dataset(FILEPATH)
zos     = ds["zos"]
nav_lat = ds["nav_lat"].values
nav_lon = ds["nav_lon"].values
nav_lon = np.where(nav_lon < 0, nav_lon + 360, nav_lon)

ds_mask   = xr.open_dataset(MASK_FILE)
umask     = ds_mask["umask"]
if "time_counter" in umask.dims:
    umask = umask.isel(time_counter=0)
depth_dim  = [d for d in umask.dims if d not in ("y", "x")][0]
ocean_mask = (umask.isel({depth_dim: 0}).values != 0)

if ocean_mask.shape != nav_lat.shape:
    print("   ⚠ Masque incompatible — ignoré")
    ocean_mask = np.ones(nav_lat.shape, dtype=bool)
else:
    print(f"   Masque OK — {ocean_mask.sum():,} pts océan")

# =============================================================================
# ÉTAPE 2 — MOYENNE TEMPORELLE + MASQUE
# =============================================================================

print("\n[2/5] Moyenne temporelle 1991–2023...")
zos_mean = zos.mean(dim="time_counter").values
zos_mean = np.where(ocean_mask, zos_mean, np.nan)

in_band = (nav_lat >= LAT_MIN) & (nav_lat <= LAT_MAX) & np.isfinite(zos_mean)
print(f"   Points valides {LAT_MIN}–{LAT_MAX}° : {in_band.sum():,}")
if in_band.sum() < 1000:
    raise RuntimeError(f"Trop peu de points ({in_band.sum()}) — vérifier masque.")

# =============================================================================
# ÉTAPE 3 — GRADIENT SSH 2D
# =============================================================================

print("\n[3/5] Calcul |∇η| (m/100 km)...")

dy = np.gradient(np.deg2rad(nav_lat), axis=0) * R_EARTH
dx = np.gradient(np.deg2rad(nav_lon), axis=1) * R_EARTH * np.cos(np.deg2rad(nav_lat))

with np.errstate(invalid="ignore", divide="ignore"):
    deta_dy = np.gradient(zos_mean, axis=0) / dy
    deta_dx = np.gradient(zos_mean, axis=1) / dx

grad_m100km = np.sqrt(deta_dx**2 + deta_dy**2) * 1e5
grad_m100km = np.where(in_band, grad_m100km, np.nan)
print(f"   |∇η| max : {np.nanmax(grad_m100km):.3f} m/100 km")

# =============================================================================
# ÉTAPE 4 — OPTIMISATION S&R PAR SECTEUR
# =============================================================================

print(f"\n[4/5] Optimisation SSH dans {N_SECTORS} secteurs...")

local_max = (grad_m100km == maximum_filter(
    np.where(np.isfinite(grad_m100km), grad_m100km, 0.0), size=5))
active = ((grad_m100km > GRAD_THRESHOLD) | local_max) & in_band

lon_centers = np.arange(LON_STEP / 2, 360, LON_STEP)
ssh_labels  = {f: [] for f in ["SAF", "PF", "SACCF"]}

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

    lo = float(np.percentile(ssh_pts, 3))
    hi = float(np.percentile(ssh_pts, 97))
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

    peak_ssh = bin_ctrs[peaks]
    best_idx = np.argsort(props["prominences"])[::-1][:3]
    top_ssh  = np.sort(peak_ssh[best_idx])[::-1]
    while len(top_ssh) < 3:
        top_ssh = np.append(top_ssh, np.nan)

    # ── Contrainte latitude PF ────────────────────────────────────────────────
    # Le SSH décroît vers le sud dans l'ACC. Le SSH moyen à PF_LAT_MAX (45°S)
    # dans ce secteur sert de seuil : tout pic PF avec SSH > seuil correspond
    # à une latitude trop au nord → on le remplace par le premier pic disponible
    # sous ce seuil.
    pts_at_pf_lat = (in_sec
                     & (nav_lat >= PF_LAT_MAX - 0.5)
                     & (nav_lat <= PF_LAT_MAX + 0.5)
                     & np.isfinite(zos_mean))
    if pts_at_pf_lat.sum() > 0:
        ssh_threshold_pf = float(np.nanmean(zos_mean[pts_at_pf_lat]))
        if not np.isnan(top_ssh[1]) and top_ssh[1] > ssh_threshold_pf:
            candidates = np.sort(peak_ssh[peak_ssh <= ssh_threshold_pf])[::-1]
            top_ssh[1] = float(candidates[0]) if len(candidates) > 0 else np.nan

    for f, zeta in zip(["SAF", "PF", "SACCF"], top_ssh):
        ssh_labels[f].append(float(zeta))

zeta_circ = {}
print("   Labels SSH circumpolaires :")
for f in ["SAF", "PF", "SACCF"]:
    vals = [v for v in ssh_labels[f] if not np.isnan(v)]
    zeta_circ[f] = float(np.median(vals)) if vals else np.nan
    info = (f"ζ = {zeta_circ[f]:.4f} m  ({len(vals)}/{N_SECTORS} sec.)"
            if vals else "aucun secteur valide")
    print(f"   {f:6s} : {info}")

# =============================================================================
# ÉTAPE 5 — INTERPOLATION SUR GRILLE RÉGULIÈRE 0.25°
# =============================================================================

print("\n[5/5] Interpolation + cartes...")

res      = 0.25
lon_reg  = np.arange(0, 360, res)
lat_reg  = np.arange(LAT_MIN, LAT_MAX + res, res)
LON_G, LAT_G = np.meshgrid(lon_reg, lat_reg)

ok     = in_band & np.isfinite(grad_m100km)
pts_xy = np.column_stack([nav_lon[ok], nav_lat[ok]])

grad_reg = griddata(pts_xy, grad_m100km[ok], (LON_G, LAT_G), method="linear")
zos_reg  = griddata(pts_xy, zos_mean[ok],    (LON_G, LAT_G), method="linear")

# =============================================================================
# HELPER — axe polaire stéréo circulaire
# =============================================================================

def make_polar_ax(fig, lat_north=-35):
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(111, projection=proj)
    ax.set_extent([-180, 180, -90, lat_north], crs=ccrs.PlateCarree())
    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor="#bbbbbb", zorder=5)
    ax.coastlines(resolution="110m", color="black", linewidth=0.7, zorder=6)
    ax.gridlines(linewidth=0.4, linestyle="--", color="grey",
                 draw_labels=False, zorder=3)
    return ax

tr = ccrs.PlateCarree()

# =============================================================================
# CARTE 1 — Fond |∇η| moyen + fronts S&R calculés (polaire stéréo)
# =============================================================================

fig1 = plt.figure(figsize=(10, 10))
ax1  = make_polar_ax(fig1, lat_north=-35)

vmax = np.nanpercentile(grad_reg, 97)
pcm  = ax1.pcolormesh(LON_G, LAT_G, grad_reg,
                      cmap="YlOrRd", vmin=0, vmax=vmax,
                      transform=tr, shading="auto",
                      rasterized=True, zorder=2)
cbar = plt.colorbar(pcm, ax=ax1, orientation="vertical", location="right",
                    fraction=0.04, pad=0.04, shrink=0.7)
cbar.set_label(r"$|\nabla\eta|$ (m 100 km$^{-1}$)", fontsize=11)

for fname, style in CALC_STYLES.items():
    zeta = zeta_circ[fname]
    if np.isnan(zeta):
        continue
    ax1.contour(LON_G, LAT_G, zos_reg,
                levels=[zeta],
                colors=[style["color"]],
                linewidths=style["lw"],
                linestyles=[style["ls"]],
                transform=tr, zorder=8)
    ax1.plot([], [], color=style["color"], lw=style["lw"], ls=style["ls"],
             label=f"{fname}  (ζ = {zeta:.3f} m)")

ax1.legend(fontsize=10, loc="lower left", framealpha=0.88,
           title="Fronts calculés (S&R 2009a)", title_fontsize=10)
ax1.set_title(
    "Fronts ACC — S&R (2009a) sur NEMO\n"
    f"Fond : $|\\nabla\\eta|$ moyen 1991–2023  ·  "
    f"{N_SECTORS} secteurs × {LON_STEP:.0f}°",
    fontsize=12, fontweight="bold", pad=18
)

out1 = os.path.join(OUTPUT_DIR, "schiavon_NEMO_SR2009a_polar_grad.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"   ✓ Carte 1 : {out1}")

# =============================================================================
# CHARGEMENT GEOJSON
# =============================================================================

lit_records, src_ls = [], {}
if os.path.exists(GEOJSON):
    with open(GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        geom  = shape(feat["geometry"])
        fname = feat["properties"].get("front",  "Unknown")
        src   = feat["properties"].get("source", "Unknown")
        lit_records.append({"front": fname, "source": src, "geom": geom})
    sources = sorted({r["source"] for r in lit_records})
    src_ls  = {s: LIT_LINESTYLES[i % len(LIT_LINESTYLES)]
               for i, s in enumerate(sources)}
    print(f"   GeoJSON : {len(lit_records)} features | {list(sources)}")
else:
    print(f"   ⚠ GeoJSON introuvable — carte 2 sans littérature")

# =============================================================================
# CARTE 2 — Fronts S&R calculés vs littérature
# Légende :
#   "Type de front" : couleur = type (lit.), noir = calculé + linestyle
#   "Source"        : linestyle par source littérature
# =============================================================================

fig2 = plt.figure(figsize=(10, 10))
ax2  = make_polar_ax(fig2, lat_north=-35)

# ── Littérature (arrière-plan) ────────────────────────────────────────────────
for r in lit_records:
    color = LIT_COLORS.get(r["front"], "gray")
    ls    = src_ls.get(r["source"], "-")
    kw    = dict(color=color, lw=LIT_LW, ls=ls, alpha=LIT_ALPHA,
                 transform=tr, zorder=3)
    geom  = r["geom"]
    if isinstance(geom, LineString):
        ax2.plot(*geom.xy, **kw)
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            ax2.plot(*line.xy, **kw)

# ── Fronts S&R calculés (premier plan, noirs) ─────────────────────────────────
for fname, style in CALC_STYLES.items():
    zeta = zeta_circ[fname]
    if np.isnan(zeta):
        continue
    ax2.contour(LON_G, LAT_G, zos_reg,
                levels=[zeta],
                colors=[style["color"]],
                linewidths=style["lw"],
                linestyles=[style["ls"]],
                transform=tr, zorder=8)

# ── Légende bloc 1 : "Type de front" ─────────────────────────────────────────
handles_type = []
for fn in ["SAF", "PF", "SACCF"]:
    c  = LIT_COLORS.get(fn, "gray")
    st = CALC_STYLES[fn]
    handles_type.append(
        mlines.Line2D([], [], color=c, lw=2.0, ls="-",
                      label=f"{fn} (littérature)"))
    handles_type.append(
        mlines.Line2D([], [], color=st["color"], lw=st["lw"], ls=st["ls"],
                      label=f"{fn} (calculé)"))

leg1 = ax2.legend(handles=handles_type,
                  loc="lower left", fontsize=8,
                  framealpha=0.88,
                  title="Type de front", title_fontsize=9)
ax2.add_artist(leg1)

# ── Légende bloc 2 : "Source" ─────────────────────────────────────────────────
if src_ls:
    handles_src = [
        mlines.Line2D([], [], color="gray", lw=1.5, ls=ls,
                      alpha=0.85, label=src)
        for src, ls in src_ls.items()
    ]
    ax2.legend(handles=handles_src,
               loc="lower right", fontsize=8,
               framealpha=0.88,
               title="Source", title_fontsize=9)

ax2.set_title(
    "Fronts ACC — S&R (2009a) NEMO vs littérature\n"
    f"Moyenne 1991–2023  ·  {N_SECTORS} secteurs × {LON_STEP:.0f}°",
    fontsize=12, fontweight="bold", pad=18
)

out2 = os.path.join(OUTPUT_DIR, "schiavon_NEMO_SR2009a_vs_lit.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"   ✓ Carte 2 : {out2}")
print("\n[DONE]")