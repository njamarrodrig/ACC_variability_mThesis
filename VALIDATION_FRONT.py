"""
compare_fronts_literature.py
=============================
Compare les fronts ACC calculés (OBS + MOD)
avec les fronts de la littérature issus d'un fichier GeoJSON.

Produit 2 cartes SouthPolarStereo sans fond de champ :
  - compare_fronts_OBS.png  (limité à -50°S)
  - compare_fronts_MOD.png  (limité à -35°S)

Les fronts calculés sont tracés en noir (épais, distincts par linestyle).
Les fronts de la littérature sont tracés en couleur semi-transparente
(rouge=SAF, bleu=PF, vert=SACCF) avec le style de ligne propre à chaque source.

Usage : python compare_fronts_literature.py
  (doit être dans le même dossier que park_acc_fronts_v3.py)
"""

# ================================================================
# imports
# ================================================================
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape, LineString, MultiLineString

# ---- import du pipeline Park ----
sys.path.insert(0, os.path.dirname(__file__))

# ================================================================
# ⚠️  CHEMIN DU MODULE — à adapter selon ton arborescence
# ================================================================
PARK_MODULE_PATH = "/cyfast/njamar/SSH/FRONT_PARK_OBSNEMO.py"
# Si le fichier est ailleurs, remplacer par le bon chemin absolu, ex :
# PARK_MODULE_PATH = "/cyfast/njamar/fronts/FRONT_PARK_OBS&NEMO.py"
# ================================================================

if not os.path.exists(PARK_MODULE_PATH):
    raise FileNotFoundError(
        f"Module Park introuvable : {PARK_MODULE_PATH}\n"
        "  → Corriger PARK_MODULE_PATH en haut de ce script.\n"
        "  → Chercher le fichier avec : "
        "find /cyfast/njamar -name 'FRONT_PARK_OBS*' 2>/dev/null"
    )

import importlib.util
_spec = importlib.util.spec_from_file_location("FRONT_PARK", PARK_MODULE_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_obs       = _mod.load_obs
load_mod       = _mod.load_mod
mask_latitudes = _mod.mask_latitudes
scan_all_levels= _mod.scan_all_levels
adapt_dp_gate  = _mod.adapt_dp_gate
find_nb_sb     = _mod.find_nb_sb
find_best_front= _mod.find_best_front
interp_front   = _mod.interp_front
LON_GRID       = _mod.LON_GRID
OBS_FILE       = _mod.OBS_FILE
MOD_FILE       = _mod.MOD_FILE

# ================================================================
# CONFIG
# ================================================================
GEOJSON_FILE = "/cyfast/njamar/method_fronts/fronts_positions.geojson"
OUTDIR       = "./outputs_park_v2"
os.makedirs(OUTDIR, exist_ok=True)

# Couleurs fronts calculés (noir, distincts par linestyle)
COMPUTED_STYLE = {
    "SAF":   {"color": "black", "lw": 2.2, "ls": "-"},
    "PF":    {"color": "black", "lw": 2.2, "ls": "--"},
    "SACCF": {"color": "black", "lw": 2.2, "ls": ":"},
}

# Couleurs fronts littérature (couleur par type, semi-transparent)
LIT_COLORS = {"SAF": "crimson", "PF": "royalblue", "SACCF": "forestgreen"}
LIT_ALPHA  = 0.45
LIT_LW     = 1.2

# Styles de ligne par source (cycle)
LIT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

LAT_MIN = -90


# ================================================================
# STEP 1 — Calcul des fronts (pipeline Park)
# ================================================================

def compute_fronts(raw, label):
    """
    Exécute le pipeline Park minimal pour obtenir SAF, PF, SACCF
    sur LON_GRID. Ne génère aucune figure ni rapport.
    """
    print(f"\n  [{label}] calcul des fronts…")
    d = mask_latitudes(raw)

    circ = scan_all_levels(d)
    if not circ:
        raise RuntimeError(f"[{label}] aucun contour circumpolaire")

    gate       = adapt_dp_gate(d)
    nb_res, sb_res = find_nb_sb(circ, gate)
    nb_lv, sb_lv   = nb_res["level"], sb_res["level"]

    saf_res, _  = find_best_front(circ, d, nb_lv, sb_lv, "north")
    pf_res,  _  = find_best_front(circ, d, nb_lv, sb_lv, "mid_north")
    sc_res,  _  = find_best_front(circ, d, nb_lv, sb_lv, "mid_south")

    fronts = {
        "SAF":   interp_front(saf_res["seg"] if saf_res else None),
        "PF":    interp_front(pf_res["seg"]  if pf_res  else None),
        "SACCF": interp_front(sc_res["seg"]  if sc_res  else None),
    }
    lat_max_data = d.get("lat_max_eff", -30)

    lv_str = (f"SAF={nb_lv if saf_res is None else saf_res['level']:.3f}  "
              f"PF={nb_lv if pf_res is None else pf_res['level']:.3f}  "
              f"SACCF={nb_lv if sc_res is None else sc_res['level']:.3f} m")
    print(f"  [{label}] niveaux : {lv_str}")
    return fronts, lat_max_data


# ================================================================
# STEP 2 — Lecture des fronts de la littérature
# ================================================================

def load_geojson(fpath):
    """
    Charge fronts_positions.geojson.
    Retourne : liste de dicts {front, source, geom}
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"GeoJSON introuvable : {fpath}")
    with open(fpath, "r", encoding="utf-8") as f:
        gj = json.load(f)

    records = []
    for feat in gj["features"]:
        geom       = shape(feat["geometry"])
        front_name = feat["properties"].get("front",  "Unknown")
        source     = feat["properties"].get("source", "Unknown")
        records.append({"front": front_name, "source": source, "geom": geom})

    sources = sorted({r["source"] for r in records})
    src_ls  = {s: LIT_LINESTYLES[i % len(LIT_LINESTYLES)]
               for i, s in enumerate(sources)}
    print(f"  [GeoJSON] {len(records)} features  |  sources : {sources}")
    return records, src_ls


# ================================================================
# STEP 3 — Carte polaire
# ================================================================

def make_polar_ax(lat_north=-35):
    """
    Crée un axe SouthPolarStereo circulaire sans fond de champ.
    Identique à la projection de référence.
    """
    proj = ccrs.SouthPolarStereo()
    ax   = plt.axes(projection=proj)
    ax.set_extent([-180, 180, -90, lat_north], crs=ccrs.PlateCarree())

    theta         = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=4)
    ax.coastlines(resolution="110m", color="black", linewidth=0.8, zorder=5)
    ax.gridlines(linewidth=0.4, linestyle="--", color="grey", draw_labels=False)

    return ax


def _plot_front_clipped(ax, lo, la_arr, lat_max_data, **kw):
    """Trace un front en masquant les portions hors du champ OBS."""
    tr = ccrs.PlateCarree()
    la = la_arr.copy().astype(float)
    la[la > lat_max_data] = np.nan
    ax.plot(lo, np.ma.masked_invalid(la), transform=tr, **kw)


def plot_lit_fronts(ax, records, src_ls):
    """Trace les fronts de la littérature (colorés, semi-transparents)."""
    tr = ccrs.PlateCarree()
    for r in records:
        color = LIT_COLORS.get(r["front"], "gray")
        ls    = src_ls.get(r["source"], "-")
        kw    = dict(color=color, lw=LIT_LW, ls=ls,
                     alpha=LIT_ALPHA, transform=tr, zorder=3)
        geom  = r["geom"]
        if isinstance(geom, LineString):
            x, y = geom.xy
            ax.plot(x, y, **kw)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                x, y = line.xy
                ax.plot(x, y, **kw)


def build_legend(ax, src_ls):
    """
    Deux blocs de légende :
      - Gauche  : type de front (couleur)
      - Droite  : source littérature (linestyle)  +  fronts calculés
    """
    tr = ccrs.PlateCarree()

    # --- Bloc 1 : types de front (couleur) ---
    handles_type = []
    for fn, col in LIT_COLORS.items():
        h = mlines.Line2D([], [], color=col, lw=2, alpha=0.9,
                          label=f"{fn} (littérature)")
        handles_type.append(h)
    for fn, st in COMPUTED_STYLE.items():
        h = mlines.Line2D([], [], color=st["color"], lw=st["lw"],
                          ls=st["ls"], label=f"{fn} (calculé)")
        handles_type.append(h)

    leg1 = ax.legend(handles=handles_type, loc="lower left",
                     fontsize=7, framealpha=0.85,
                     title="Type de front", title_fontsize=8)
    ax.add_artist(leg1)

    # --- Bloc 2 : sources littérature (linestyle) ---
    handles_src = []
    for src, ls in src_ls.items():
        h = mlines.Line2D([], [], color="gray", lw=1.5, ls=ls,
                          alpha=0.85, label=src)
        handles_src.append(h)
    ax.legend(handles=handles_src, loc="lower right",
              fontsize=7, framealpha=0.85,
              title="Source", title_fontsize=8)


def make_map(label, fronts, lat_max_data, lit_records, src_ls,
             lat_north, title_suffix=""):
    """Génère et sauvegarde une carte."""
    fig = plt.figure(figsize=(10, 10))
    ax  = make_polar_ax(lat_north=lat_north)

    is_obs = (label.upper() == "OBS")
    tr     = ccrs.PlateCarree()

    # --- Fronts de la littérature (arrière-plan) ---
    plot_lit_fronts(ax, lit_records, src_ls)

    # --- Fronts calculés (premier plan) ---
    for fn, st in COMPUTED_STYLE.items():
        la_arr = fronts.get(fn)
        if la_arr is None or not np.any(np.isfinite(la_arr)):
            continue
        if is_obs:
            _plot_front_clipped(ax, LON_GRID, la_arr, lat_max_data,
                                color=st["color"], lw=st["lw"], ls=st["ls"],
                                zorder=6)
        else:
            ok = np.isfinite(la_arr)
            ax.plot(LON_GRID[ok], la_arr[ok], transform=tr,
                    color=st["color"], lw=st["lw"], ls=st["ls"], zorder=6)

    # --- Légendes ---
    build_legend(ax, src_ls)

    ax.set_title(
        f"Fronts ACC – {label}  (calculés vs littérature)\n{title_suffix}",
        fontsize=12, fontweight="bold", pad=20
    )

    out = os.path.join(OUTDIR, f"compare_fronts_{label}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "=" * 60)
    print("  compare_fronts_literature.py")
    print("  Fronts calculés (Park 2019) vs littérature (GeoJSON)")
    print("=" * 60)

    # 1. Chargement et calcul des fronts
    fronts_obs, lat_max_obs = None, -50
    fronts_mod, lat_max_mod = None, -35

    try:
        obs = load_obs(OBS_FILE)
        fronts_obs, lat_max_obs = compute_fronts(obs, "OBS")
    except FileNotFoundError as e:
        print(f"[SKIP OBS] {e}")

    try:
        mod = load_mod(MOD_FILE)
        fronts_mod, lat_max_mod = compute_fronts(mod, "MOD")
    except FileNotFoundError as e:
        print(f"[SKIP MOD] {e}")

    # 2. Fronts de la littérature
    try:
        lit_records, src_ls = load_geojson(GEOJSON_FILE)
    except FileNotFoundError as e:
        print(f"[ERREUR GeoJSON] {e}")
        print(f"  Vérifier que le fichier existe : {GEOJSON_FILE}")
        print(f"  Les cartes seront générées sans fronts de littérature.")
        lit_records, src_ls = [], {}

    # 3. Cartes
    if fronts_obs is not None:
        make_map("OBS", fronts_obs, lat_max_obs,
                 lit_records, src_ls,
                 lat_north=-50,
                 title_suffix="OBS limité à 50°S")

    if fronts_mod is not None:
        make_map("MOD", fronts_mod, lat_max_mod,
                 lit_records, src_ls,
                 lat_north=-35,
                 title_suffix="MOD — 35°S à 90°S")

    print(f"\n[DONE] Sorties dans : {OUTDIR}/")


if __name__ == "__main__":
    main()
