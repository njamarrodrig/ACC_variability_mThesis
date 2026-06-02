"""
compare_fronts_literature.py
=============================
Compare les fronts ACC calculés (OBS + MOD)
avec les fronts de la littérature issus d'un fichier GeoJSON.

Produit :
  - 2 cartes SouthPolarStereo (overlay visuel) :
      compare_fronts_OBS.png   (limité à -50°S)
      compare_fronts_MOD.png   (limité à -35°S)
  - 1 évaluation QUANTITATIVE :
      fronts_metrics.csv       : tableau latitudes moyennes / biais / r

Évaluation quantitative
-----------------------
Chaque front de la littérature est rééchantillonné sur LON_GRID (la grille
de longitudes des fronts calculés). Pour chaque longitude, on dispose alors
d'un couple de latitudes comparables. Sur les longitudes COMMUNES aux deux
trajectoires, on calcule :
  - lat_calc / lat_lit : latitudes moyennes des deux trajectoires
  - biais   : <phi_calc - phi_lit>      décalage méridien systématique
              (biais > 0  =>  front calculé plus au NORD)
  - r       : corrélation de Pearson entre phi_calc(lon) et phi_lit(lon)
              => accord des excursions méridiennes le long du courant

Ce n'est PAS une validation stricte : données, périodes et critères de
définition des fronts diffèrent. Les métriques mesurent un écart / une
cohérence entre trajectoires, pas une erreur par rapport à une vérité terrain.

Usage : python compare_fronts_literature.py
  (doit être dans le même dossier que park_acc_fronts_v3.py)
"""

# ================================================================
# imports
# ================================================================
import os
import sys
import csv
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
# CHEMIN DU MODULE - a adapter selon ton arborescence
# ================================================================
PARK_MODULE_PATH = "/cyfast/njamar/SSH/FRONT_PARK_OBSNEMO.py"
# Si le fichier est ailleurs, remplacer par le bon chemin absolu, ex :
# PARK_MODULE_PATH = "/cyfast/njamar/fronts/FRONT_PARK_OBS&NEMO.py"
# ================================================================

if not os.path.exists(PARK_MODULE_PATH):
    raise FileNotFoundError(
        f"Module Park introuvable : {PARK_MODULE_PATH}\n"
        "  -> Corriger PARK_MODULE_PATH en haut de ce script.\n"
        "  -> Chercher le fichier avec : "
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
LON_GRID       = np.asarray(_mod.LON_GRID, dtype=float)
OBS_FILE       = _mod.OBS_FILE
MOD_FILE       = _mod.MOD_FILE

# ================================================================
# CONFIG
# ================================================================
GEOJSON_FILE = "/cyfast/njamar/method_fronts/fronts_positions.geojson"
OUTDIR       = "./outputs_park_v2"
os.makedirs(OUTDIR, exist_ok=True)

METRICS_CSV  = os.path.join(OUTDIR, "fronts_metrics.csv")

# Couleurs fronts calculés (noir, distincts par linestyle)
COMPUTED_STYLE = {
    "SAF":   {"color": "black", "lw": 2.2, "ls": "-"},
    "PF":    {"color": "black", "lw": 2.2, "ls": "--"},
    "SACCF": {"color": "black", "lw": 2.2, "ls": ":"},
}

FRONT_TYPES = ["SAF", "PF", "SACCF"]

# Couleurs fronts littérature (couleur par type, semi-transparent)
LIT_COLORS = {"SAF": "crimson", "PF": "royalblue", "SACCF": "forestgreen"}
LIT_ALPHA  = 0.45
LIT_LW     = 1.2

# Styles de ligne par source (cycle)
LIT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

LAT_MIN = -90

# Nombre minimal de longitudes communes pour qu'une métrique soit jugée fiable
MIN_COMMON_PTS = 5


# ================================================================
# STEP 1 - Calcul des fronts (pipeline Park)
# ================================================================

def compute_fronts(raw, label):
    """
    Exécute le pipeline Park minimal pour obtenir SAF, PF, SACCF
    sur LON_GRID. Ne génère aucune figure ni rapport.
    """
    print(f"\n  [{label}] calcul des fronts...")
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
# STEP 2 - Lecture des fronts de la littérature
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


def _norm_front(name):
    """Normalise un nom de front pour la correspondance (SAF/PF/SACCF)."""
    return str(name).upper().strip().replace("-", "").replace("_", "").replace(" ", "")


# ================================================================
# STEP 3 - Carte polaire
# ================================================================

def make_polar_ax(lat_north=-35):
    """Crée un axe SouthPolarStereo circulaire sans fond de champ."""
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
    """Deux blocs de légende : type de front (couleur) + source (linestyle)."""
    handles_type = []
    for fn, col in LIT_COLORS.items():
        handles_type.append(mlines.Line2D([], [], color=col, lw=2, alpha=0.9,
                                          label=f"{fn} (littérature)"))
    for fn, st in COMPUTED_STYLE.items():
        handles_type.append(mlines.Line2D([], [], color=st["color"], lw=st["lw"],
                                          ls=st["ls"], label=f"{fn} (calculé)"))
    leg1 = ax.legend(handles=handles_type, loc="lower left",
                     fontsize=14, framealpha=0.85,
                     title="Type de front", title_fontsize=16)
    ax.add_artist(leg1)

    handles_src = [mlines.Line2D([], [], color="gray", lw=1.5, ls=ls,
                                 alpha=0.85, label=src)
                   for src, ls in src_ls.items()]
    ax.legend(handles=handles_src, loc="lower right",
              fontsize=14, framealpha=0.85,
              title="Source", title_fontsize=16)


def make_map(label, fronts, lat_max_data, lit_records, src_ls,
             lat_north, title_suffix=""):
    """Génère et sauvegarde une carte polaire (overlay visuel)."""
    fig = plt.figure(figsize=(10, 10))
    ax  = make_polar_ax(lat_north=lat_north)
    is_obs = (label.upper() == "OBS")
    tr     = ccrs.PlateCarree()

    plot_lit_fronts(ax, lit_records, src_ls)

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

    if not is_obs:
        build_legend(ax, src_ls)
    ax.set_title(
        f"Fronts ACC - {label}  (calculés vs littérature)\n{title_suffix}",
        fontsize=12, fontweight="bold", pad=20)

    out = os.path.join(OUTDIR, f"compare_fronts_{label}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out}")


# ================================================================
# STEP 4 - EVALUATION QUANTITATIVE  (NOUVEAU)
# ================================================================

def densify_geom(geom, n_points=6000):
    """
    Échantillonne une LineString / MultiLineString le long de son arc
    (pas le long de la longitude, pour gérer correctement les méandres).
    Retourne deux tableaux (lon, lat).
    """
    if isinstance(geom, LineString):
        lines = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        return np.array([]), np.array([])

    lengths = [ln.length for ln in lines]
    total   = sum(L for L in lengths if L > 0)
    if total <= 0:
        return np.array([]), np.array([])

    lons, lats = [], []
    for ln, L in zip(lines, lengths):
        if L <= 0:
            continue
        npts = max(int(n_points * L / total), 2)
        for dist in np.linspace(0.0, L, npts):
            p = ln.interpolate(dist)
            lons.append(p.x)
            lats.append(p.y)
    return np.asarray(lons, float), np.asarray(lats, float)


def resample_to_grid(lons, lats, lon_grid):
    """
    Rééchantillonne un nuage (lon, lat) sur lon_grid.
    Pour chaque longitude de la grille, moyenne des latitudes tombant
    dans la classe correspondante. NaN si aucune donnée.
    Gère les conventions [0,360] et [-180,180] ainsi qu'une grille
    non triée.
    """
    lon_grid = np.asarray(lon_grid, float)
    out = np.full(lon_grid.shape, np.nan)
    if lons.size == 0:
        return out

    # convention de longitude alignée sur LON_GRID
    if np.nanmin(lon_grid) < 0.0:
        lons = ((lons + 180.0) % 360.0) - 180.0
    else:
        lons = lons % 360.0

    # grille triée + bords de classe
    order  = np.argsort(lon_grid)
    lon_s  = lon_grid[order]
    edges  = np.empty(lon_s.size + 1)
    edges[1:-1] = 0.5 * (lon_s[:-1] + lon_s[1:])
    edges[0]    = lon_s[0]  - 0.5 * (lon_s[1]  - lon_s[0])
    edges[-1]   = lon_s[-1] + 0.5 * (lon_s[-1] - lon_s[-2])

    idx   = np.digitize(lons, edges) - 1
    valid = (idx >= 0) & (idx < lon_s.size) & np.isfinite(lats)
    idx_v, lat_v = idx[valid], lats[valid]
    if idx_v.size:
        sums  = np.bincount(idx_v, weights=lat_v, minlength=lon_s.size)
        cnts  = np.bincount(idx_v, minlength=lon_s.size)
        out_s = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        out[order] = out_s
    return out


def lit_fronts_on_grid(records, lon_grid):
    """
    Regroupe les fronts de la littérature par (source, type de front)
    et les rééchantillonne sur lon_grid.
    Retourne : dict {(source, FRONT): latitude_array}
    """
    groups = {}
    for r in records:
        fn = _norm_front(r["front"])
        if fn not in FRONT_TYPES:
            continue  # STF, SB, etc. : non comparés
        groups.setdefault((r["source"], fn), []).append(r["geom"])

    out = {}
    for (src, fn), geoms in groups.items():
        lo_all, la_all = [], []
        for g in geoms:
            lo, la = densify_geom(g)
            if lo.size:
                lo_all.append(lo)
                la_all.append(la)
        if not lo_all:
            continue
        lo = np.concatenate(lo_all)
        la = np.concatenate(la_all)
        out[(src, fn)] = resample_to_grid(lo, la, lon_grid)
    return out


def compute_metrics(lat_comp, lat_lit):
    """
    Calcule les descripteurs quantitatifs sur les longitudes communes :
    latitudes moyennes, biais et corrélation.
    Convention de biais : lat_comp - lat_lit
      (biais > 0  =>  front calculé plus au NORD que la référence).
    """
    c = np.asarray(lat_comp, float)
    l = np.asarray(lat_lit,  float)
    n = min(c.size, l.size)
    c, l = c[:n], l[:n]
    ok = np.isfinite(c) & np.isfinite(l)
    n_ok = int(ok.sum())

    res = dict(n=n_ok, mean_comp=np.nan, mean_lit=np.nan,
               bias=np.nan, r=np.nan)
    if n_ok < 3:
        return res

    c, l = c[ok], l[ok]
    bias = float(np.mean(c - l))
    sc, sl = float(np.std(c)), float(np.std(l))
    r = float(np.corrcoef(c, l)[0, 1]) if (sc > 1e-9 and sl > 1e-9) else np.nan

    res.update(mean_comp=float(np.mean(c)), mean_lit=float(np.mean(l)),
               bias=bias, r=r)
    return res


def quantitative_comparison(label, fronts, lat_max_data, lit_on_grid, is_obs):
    """
    Compare les fronts calculés aux fronts de la littérature.
    Retourne une liste de dicts (une ligne par couple front x source).
    """
    rows = []
    for fn in FRONT_TYPES:
        comp = fronts.get(fn)
        if comp is None:
            continue
        comp = np.asarray(comp, float).copy()
        if comp.size != LON_GRID.size:
            print(f"  [WARN] {label}/{fn} : taille incohérente avec LON_GRID, ignoré")
            continue
        # OBS : on restreint le front calculé au champ valide (comme la carte)
        if is_obs:
            comp[comp > lat_max_data] = np.nan

        for (src, lit_fn), lit_lat in sorted(lit_on_grid.items()):
            if lit_fn != fn:
                continue
            m = compute_metrics(comp, lit_lat)
            m.update(dataset=label, front=fn, source=src)
            rows.append(m)
    return rows


def print_metrics_table(rows):
    """Affiche le tableau de métriques dans la console."""
    if not rows:
        print("  (aucune métrique : pas de fronts de littérature comparables)")
        return
    hdr = (f"{'dataset':<8}{'front':<7}{'source':<24}{'N':>5}"
           f"{'lat_calc':>10}{'lat_lit':>10}{'biais':>9}{'r':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in rows:
        flag = "" if m["n"] >= MIN_COMMON_PTS else "  (N faible)"
        def f(x, d=2):
            return f"{x:.{d}f}" if np.isfinite(x) else "  --"
        print(f"{m['dataset']:<8}{m['front']:<7}{m['source']:<24}{m['n']:>5}"
              f"{f(m['mean_comp']):>10}{f(m['mean_lit']):>10}{f(m['bias']):>9}"
              f"{f(m['r'],3):>8}{flag}")
    print("-" * len(hdr))
    print("  biais = lat_calc - lat_lit  (>0 : front calculé plus au nord)")
    print("  unités : degrés de latitude  |  r : corrélation lat(lon)")


def write_metrics_csv(rows, fpath):
    """Écrit le tableau de métriques au format CSV."""
    cols = ["dataset", "front", "source", "n",
            "mean_comp", "mean_lit", "bias", "r"]
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in rows:
            w.writerow({k: m.get(k, "") for k in cols})
    print(f"[CSV] {fpath}")


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

    # 3. Cartes polaires (overlay visuel)
    if fronts_obs is not None:
        make_map("OBS", fronts_obs, lat_max_obs, lit_records, src_ls,
                 lat_north=-50, title_suffix="OBS limité à 50°S")
    if fronts_mod is not None:
        make_map("MOD", fronts_mod, lat_max_mod, lit_records, src_ls,
                 lat_north=-35, title_suffix="MOD - 35°S à 90°S")

    # 4. EVALUATION QUANTITATIVE
    print("\n" + "=" * 60)
    print("  ÉVALUATION QUANTITATIVE")
    print("=" * 60)

    if not lit_records:
        print("  Pas de fronts de littérature : évaluation quantitative ignorée.")
        print(f"\n[DONE] Sorties dans : {OUTDIR}/")
        return

    # rééchantillonnage des fronts de littérature sur LON_GRID
    lit_on_grid = lit_fronts_on_grid(lit_records, LON_GRID)
    print(f"  Fronts de littérature rééchantillonnés : "
          f"{len(lit_on_grid)} couples (source, front)")

    all_rows = []
    if fronts_obs is not None:
        all_rows += quantitative_comparison("OBS", fronts_obs, lat_max_obs,
                                            lit_on_grid, is_obs=True)
    if fronts_mod is not None:
        all_rows += quantitative_comparison("MOD", fronts_mod, lat_max_mod,
                                            lit_on_grid, is_obs=False)

    print_metrics_table(all_rows)
    write_metrics_csv(all_rows, METRICS_CSV)

    print(f"\n[DONE] Sorties dans : {OUTDIR}/")


if __name__ == "__main__":
    main()