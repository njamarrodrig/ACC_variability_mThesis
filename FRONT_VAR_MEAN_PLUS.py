"""
ACC_FRONT_METHODS.py
=====================
Pipeline methodologique complet :
  Etape A — Methode fixe (Kim & Orsi 2014)          : niveaux SSH fixes
  Etape B — Methode annuelle                          : niveaux SSH recalibres par an
  Etape C — Comparaison fixe vs annuel               : biais de la methode fixe
  Etape D — Validation par gradient SSH              : coherence contour / jet
  Etape E — Validation par vitesse geostrophique     : OBS uniquement (si ug/vg dispo)
  Etape F — Desaisonnalisation                       : tendances sur anomalies

FIGURES PRODUITES :
  fig8_fixed_<PROD>.png         — Fig. 8 Kim & Orsi, niveaux fixes
  fig8_annual_<PROD>.png        — Fig. 8, niveaux annuels
  fig8_fixed_deseas_<PROD>.png  — Fig. 8, niveaux fixes + desaisonnalise
  fig8_annual_deseas_<PROD>.png — Fig. 8, niveaux annuels + desaisonnalise
  fig_bias_<PROD>.png           — Biais fixe - annuel par front
  fig_gradient_val_<PROD>.png   — Distance contour / max gradient SSH
  fig_speed_val_OBS.png         — Distance contour / max vitesse (OBS seul)
  annual_levels_<PROD>.csv      — Niveaux SSH annuels retenus
  table_annual_<PROD>.csv       — Positions et deplacements annuels

Niveaux SSH de reference (= FRONT_PARK_OBSNEMO.py) :
  OBS  -> SAF = -0.980 m  |  PF = -1.340 m  |  SACCF = -1.720 m
  NEMO -> SAF = -0.620 m  |  PF = -0.960 m  |  SACCF = -1.280 m

Usage : python ACC_FRONT_METHODS.py
"""

import os, csv, warnings
warnings.filterwarnings("ignore")

import numpy as np
import numpy.ma as ma
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OBS_FILE = "dot_all_30bmedian_eigen6s4v2_sig3.nc"
MOD_FILE = "zos_1991_2023.nc"
OUTDIR   = "./outputs_var_mean_nobiais"
os.makedirs(OUTDIR, exist_ok=True)

OBS_T0, OBS_T1 = "2002-07", "2018-10"
MOD_T0, MOD_T1 = "1991-01", "2023-12"

# Niveaux SSH de reference (methode fixe, identique a FRONT_PARK_OBSNEMO.py)
REF_LEVELS = {
    "OBS":  {"SAF": -0.980, "PF": -1.340, "SACCF": -1.720},
    "NEMO": {"SAF": -0.620, "PF": -0.960, "SACCF": -1.280},
}

FRONT_NAMES  = ["SAF", "PF", "SACCF"]
FRONT_LABELS = {
    "SAF":   "Sub-Antarctic Front (SAF)",
    "PF":    "Polar Front (PF)",
    "SACCF": "Southern ACC Front (sACCf)",
}
FRONT_COLORS = {"SAF": "#1f78b4", "PF": "#33a02c", "SACCF": "#e31a1c"}

LON_GRID        = np.arange(-180, 181, 1.0)
MIN_SEGMENT_PTS = 30
SEARCH_BAND_DEG = 1.5
MIN_COVERAGE    = 0.70
LOWPASS_MONTHS  = 12

# Paramètres pour le recalibrage annuel des niveaux SSH (Etape B)
SEARCH_LEVEL_RANGE = 0.15   # m : plage de recherche autour du niveau fixe
SEARCH_LEVEL_STEP  = 0.01   # m : pas de recherche

# Ponderation du score de selection du niveau annuel
W_GRADIENT = 0.6   # poids du gradient SSH le long du contour
W_COVERAGE = 0.3   # poids de la couverture zonale
W_DEVIATION = 0.1  # penalite pour s'eloigner du niveau de reference

# Bande de recherche du maximum de gradient / vitesse (Etape D/E)
GRADIENT_SEARCH_DEG = 2.0   # +/- degrees autour du contour


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def remap_curvilinear(lon, lat, field):
    nx = lon.shape[0]
    nlo = np.zeros_like(lon); nla = np.zeros_like(lat)
    nfi = np.full_like(field, np.nan, dtype=float)
    for i in range(nx):
        j = int(np.argmin(lon[i, :]))
        nlo[i, :] = np.roll(lon[i, :],   -j)
        nla[i, :] = np.roll(lat[i, :],   -j)
        nfi[i, :] = np.roll(field[i, :], -j)
    nlo = np.where(nlo > 180, nlo - 360, nlo)
    return nlo, nla, nfi


def load_obs(fpath, t0, t1):
    ds  = xr.open_dataset(fpath)
    dot = ds["dot"].sel(time=slice(t0, t1))
    if "land_mask" in ds:
        dot = dot.where(ds["land_mask"] == 0)
    dot = dot.transpose("time", "latitude", "longitude")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon = np.where(lon > 180, lon - 360, lon)
    idx = np.argsort(lon); lon = lon[idx]
    field      = dot.values[:, :, idx]
    LO, LA     = np.meshgrid(lon, lat)
    time       = dot["time"].values
    mean_field = np.nanmean(field, axis=0)
    # Champs de vitesse geostrophique si disponibles
    ug_all = None; vg_all = None
    if "ug" in ds and "vg" in ds:
        ug = ds["ug"].sel(time=slice(t0, t1)).transpose(
            "time", "latitude", "longitude").values[:, :, idx]
        vg = ds["vg"].sel(time=slice(t0, t1)).transpose(
            "time", "latitude", "longitude").values[:, :, idx]
        ug_all = ug; vg_all = vg
        print(f"[OBS]  vitesse geostrophique disponible (ug, vg)")
    print(f"[OBS]  {len(time)} mois  ({str(time[0])[:7]} -> {str(time[-1])[:7]})")
    return dict(name="OBS", time=time, lon2d=LO, lat2d=LA,
                field=field, mean_field=mean_field,
                ug=ug_all, vg=vg_all)


def load_nemo(fpath, t0, t1):
    ds  = xr.open_dataset(fpath)
    zos = ds["zos"]
    if "time_counter" in zos.dims:
        zos = zos.rename({"time_counter": "time"})
    zos = zos.sel(time=slice(t0, t1)).where(lambda x: x != 0)
    sdims = [d for d in zos.dims if d != "time"]
    zos   = zos.transpose("time", *sdims)
    time  = zos["time"].values
    T     = len(time)
    lon_r = ds["nav_lon"].values
    lat_r = ds["nav_lat"].values
    fl = []
    for t in range(T):
        _, _, fi = remap_curvilinear(lon_r, lat_r, zos.values[t].astype(float))
        fl.append(fi)
    lo2, la2, _ = remap_curvilinear(lon_r, lat_r, zos.values[0].astype(float))
    field  = np.stack(fl, axis=0)
    mf_raw = remap_curvilinear(
        lon_r, lat_r, np.nanmean(zos.values, axis=0).astype(float))[2]
    print(f"[NEMO] {len(time)} mois  ({str(time[0])[:7]} -> {str(time[-1])[:7]})")
    return dict(name="NEMO", time=time, lon2d=lo2, lat2d=la2,
                field=field, mean_field=mf_raw, ug=None, vg=None)


# ─────────────────────────────────────────────
# GRADIENT SSH
# ─────────────────────────────────────────────

def compute_ssh_gradient(lon2d, lat2d, field2d):
    """
    Calcule |nabla SSH| en m/deg (approximation sur grille reguliere).

    |nabla SSH| = sqrt( (dSSH/dlon)^2 + (dSSH/dlat)^2 )

    Utilise np.gradient qui applique des differences finies centrees
    (2eme ordre) sur l'interieur et des differences decentrees aux bords.
    """
    # Espacement en degres
    dlat = np.gradient(lat2d, axis=0)   # variation de lat selon l'axe 0
    dlon = np.gradient(lon2d, axis=1)   # variation de lon selon l'axe 1

    # Gradient SSH
    dSSH_dlat = np.gradient(field2d, axis=0) / np.where(np.abs(dlat) > 1e-6,
                                                          dlat, np.nan)
    dSSH_dlon = np.gradient(field2d, axis=1) / np.where(np.abs(dlon) > 1e-6,
                                                          dlon, np.nan)
    return np.sqrt(dSSH_dlat**2 + dSSH_dlon**2)


# ─────────────────────────────────────────────
# EXTRACTION DES CONTOURS
# ─────────────────────────────────────────────

def extract_contour(lon2d, lat2d, field2d, level):
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(lon2d, lat2d, field2d, levels=[level])
    except Exception:
        plt.close(fig); return []
    plt.close(fig)
    try:
        paths = cs.get_paths()
    except AttributeError:
        paths = cs.collections[0].get_paths() if cs.collections else []
    segs = []
    for path in paths:
        v = path.vertices
        lo, la = v[:, 0], v[:, 1]
        ok = np.isfinite(lo) & np.isfinite(la)
        lo, la = lo[ok], la[ok]
        if len(lo) >= MIN_SEGMENT_PTS:
            segs.append({"lon": lo, "lat": la})
    return segs


def score_segment(seg, grad2d, lon2d, lat2d,
                  ref_level, level, band=1.0):
    """
    Score composite d'un segment de contour :
      G = gradient SSH moyen le long du segment
      C = couverture zonale normalisee [0,1]
      D = penalite pour ecart au niveau de reference

    score = W_GRADIENT*G_norm + W_COVERAGE*C - W_DEVIATION*D_norm
    """
    lo, la = seg["lon"], seg["lat"]

    # Couverture zonale
    cov = (np.nanmax(lo) - np.nanmin(lo)) / 360.0

    # Gradient SSH moyen le long du contour
    g_vals = []
    for xi, yi in zip(lo, la):
        msk = (np.abs(lon2d - xi) <= band) & (np.abs(lat2d - yi) <= band)
        if np.any(msk):
            v = np.nanmean(grad2d[msk])
            if np.isfinite(v):
                g_vals.append(v)
    g_mean = np.nanmean(g_vals) if g_vals else 0.0

    # Penalite de deviation par rapport au niveau de reference
    d_pen = abs(level - ref_level) / (SEARCH_LEVEL_RANGE + 1e-9)

    return g_mean, cov, d_pen


def select_main_branch_gradient(segs, grad2d, lon2d, lat2d,
                                 ref_level, level,
                                 clim_lat=None, prev_lat=None):
    """
    Selectionne la branche principale en combinant :
      - gradient SSH (Sokolov & Rintoul : front = maximum de gradient)
      - couverture zonale
      - penalite de deviation du niveau de reference
      - coherence avec la climatologie et le mois precedent
    """
    if not segs:
        return None, 0.0
    best_seg, best_score = None, -np.inf
    for seg in segs:
        lo, la = seg["lon"], seg["lat"]
        g_mean, cov, d_pen = score_segment(
            seg, grad2d, lon2d, lat2d, ref_level, level)

        # Coherence temporelle (comme avant)
        dc, dp = 0.0, 0.0
        if clim_lat is not None:
            idx = np.array([np.argmin(np.abs(LON_GRID - x)) for x in lo])
            ref = clim_lat[idx]; v = np.isfinite(ref)
            if np.any(v):
                dc = np.nanmean(np.abs(la[v] - ref[v]))
        if prev_lat is not None:
            idx = np.array([np.argmin(np.abs(LON_GRID - x)) for x in lo])
            ref = prev_lat[idx]; v = np.isfinite(ref)
            if np.any(v):
                dp = np.nanmean(np.abs(la[v] - ref[v]))

        # Normalisation du gradient (ordre de grandeur ~ 0.01 m/deg)
        g_norm = min(g_mean / 0.02, 1.0)

        score = (W_GRADIENT * g_norm
                 + W_COVERAGE * cov
                 - W_DEVIATION * d_pen
                 - 0.05 * dc - 0.03 * dp)

        if score > best_score:
            best_score = score
            best_seg   = seg
    return best_seg, best_score


def interp_to_lon_grid(seg):
    if seg is None:
        return np.full(len(LON_GRID), np.nan)
    lo, la = seg["lon"], seg["lat"]
    idx = np.argsort(lo); lo, la = lo[idx], la[idx]
    out = np.full(len(LON_GRID), np.nan)
    for i, x in enumerate(LON_GRID):
        sel = np.abs(lo - x) <= SEARCH_BAND_DEG
        if np.any(sel):
            out[i] = np.nanmedian(la[sel])
    return out


def apply_lat_bounds(lat_arr, fn):
    bounds = {"SAF": (-60, -38), "PF": (-65, -45), "SACCF": (-70, -52)}
    lat_s, lat_n = bounds[fn]
    out = lat_arr.copy()
    out[(out < lat_s) | (out > lat_n)] = np.nan
    return out


# ─────────────────────────────────────────────
# ETAPE A — METHODE FIXE (Kim & Orsi 2014)
# ─────────────────────────────────────────────

def extract_phi_fixed(data, ref_levels, prod_name):
    """
    Methode A : niveaux SSH fixes sur toute la periode.
    Identique a Kim & Orsi (2014).
    Le gradient SSH est utilise pour la selection de branche
    mais le niveau lui-meme ne change pas d'une annee a l'autre.
    """
    T    = data["field"].shape[0]
    lo2  = data["lon2d"]; la2 = data["lat2d"]
    time = data["time"]

    # Reference initiale = champ moyen
    mean_grad = compute_ssh_gradient(lo2, la2, data["mean_field"])
    clim_ref  = {}
    for fn in FRONT_NAMES:
        segs = extract_contour(lo2, la2, data["mean_field"], ref_levels[fn])
        seg, _ = select_main_branch_gradient(
            segs, mean_grad, lo2, la2, ref_levels[fn], ref_levels[fn])
        clim_ref[fn] = apply_lat_bounds(interp_to_lon_grid(seg), fn)

    phi    = {fn: np.full((T, len(LON_GRID)), np.nan) for fn in FRONT_NAMES}
    prev   = {fn: clim_ref[fn].copy() for fn in FRONT_NAMES}

    print(f"  [Methode A - {prod_name}] niveaux fixes, {T} mois...",
          end="", flush=True)
    for t in range(T):
        if t % 24 == 0: print(f" {t}", end="", flush=True)
        f2d  = data["field"][t]
        grad = compute_ssh_gradient(lo2, la2, f2d)
        for fn in FRONT_NAMES:
            segs  = extract_contour(lo2, la2, f2d, ref_levels[fn])
            seg, _ = select_main_branch_gradient(
                segs, grad, lo2, la2, ref_levels[fn], ref_levels[fn],
                clim_lat=clim_ref[fn], prev_lat=prev[fn])
            lat_t = apply_lat_bounds(interp_to_lon_grid(seg), fn)
            phi[fn][t] = lat_t
            prev[fn] = np.where(np.isfinite(lat_t), lat_t, prev[fn])
    print("  OK")
    return phi


# ─────────────────────────────────────────────
# ETAPE B — METHODE ANNUELLE
# ─────────────────────────────────────────────

def estimate_annual_front_levels(data, ref_levels, prod_name):
    """
    Methode B : recalibrage annuel du niveau SSH de chaque front.

    Pour chaque annee Y :
      1. Calcul du champ moyen annuel eta_bar(x,y,Y)
      2. Pour chaque front f, balayage des niveaux candidats autour
         du niveau de reference eta_f_ref +/- SEARCH_LEVEL_RANGE
      3. Pour chaque niveau candidat, extraction du contour et calcul
         du score : gradient SSH + couverture - deviation
      4. Retention du niveau avec le meilleur score

    Justification :
      Le niveau moyen de la mer peut varier d'une annee a l'autre
      (cycle stérique, tendance de hausse du niveau marin, forçage
      atmospherique). Un niveau SSH fixe peut done se deplacer
      meridionalement uniquement parce que le fond SSH a derive,
      sans que le jet lui-meme ait bouge.
      Recalibrer annuellement le niveau corrige ce biais de premiere ordre.

    Retourne :
      annual_levels : dict[year][fn] -> float (niveau SSH retenu)
      level_records : liste de dicts pour export CSV
    """
    time      = data["time"]
    years_all = np.array([int(str(t)[:4]) for t in time])
    unique_yrs = np.unique(years_all)
    lo2 = data["lon2d"]; la2 = data["lat2d"]

    annual_levels = {}
    level_records = []

    print(f"  [Methode B - {prod_name}] recalibrage annuel "
          f"({len(unique_yrs)} annees)...")

    for y in unique_yrs:
        idx_y = years_all == y
        n_y   = int(np.sum(idx_y))
        if n_y < 6:
            # Annee incomplete : garder le niveau de reference
            annual_levels[y] = {fn: ref_levels[fn] for fn in FRONT_NAMES}
            for fn in FRONT_NAMES:
                level_records.append({
                    "year": y, "front": fn,
                    "level_fixed": ref_levels[fn],
                    "level_annual": ref_levels[fn],
                    "delta_level": 0.0,
                    "n_months": n_y,
                    "score": np.nan,
                    "note": "annee_incomplete",
                })
            continue

        # Champ moyen annuel
        annual_mean = np.nanmean(data["field"][idx_y], axis=0)
        grad_annual = compute_ssh_gradient(lo2, la2, annual_mean)

        annual_levels[y] = {}
        for fn in FRONT_NAMES:
            ref_lv   = ref_levels[fn]
            best_lv  = ref_lv
            best_sc  = -np.inf

            # Balayage des niveaux candidats
            candidates = np.arange(
                ref_lv - SEARCH_LEVEL_RANGE,
                ref_lv + SEARCH_LEVEL_RANGE + SEARCH_LEVEL_STEP * 0.5,
                SEARCH_LEVEL_STEP
            )
            for lv in candidates:
                segs = extract_contour(lo2, la2, annual_mean, lv)
                if not segs:
                    continue
                # On prend le meilleur segment pour ce niveau
                best_seg_lv = None; sc_lv = -np.inf
                for seg in segs:
                    g, c, d = score_segment(
                        seg, grad_annual, lo2, la2, ref_lv, lv)
                    g_norm = min(g / 0.02, 1.0)
                    cov_n  = (np.nanmax(seg["lon"]) - np.nanmin(seg["lon"]))/360
                    sc = (W_GRADIENT * g_norm
                          + W_COVERAGE * cov_n
                          - W_DEVIATION * abs(lv - ref_lv) /
                          (SEARCH_LEVEL_RANGE + 1e-9))
                    if sc > sc_lv:
                        sc_lv = sc; best_seg_lv = seg
                if best_seg_lv is not None and sc_lv > best_sc:
                    best_sc = sc_lv
                    best_lv = lv

            annual_levels[y][fn] = best_lv
            level_records.append({
                "year": y, "front": fn,
                "level_fixed": ref_lv,
                "level_annual": best_lv,
                "delta_level": best_lv - ref_lv,
                "n_months": n_y,
                "score": best_sc,
                "note": "",
            })

        print(f"    {y} : SAF={annual_levels[y]['SAF']:.3f}  "
              f"PF={annual_levels[y]['PF']:.3f}  "
              f"SACCF={annual_levels[y]['SACCF']:.3f}")

    return annual_levels, level_records


def extract_phi_annual(data, annual_levels, ref_levels, prod_name):
    """
    Methode B : extraction des contours avec les niveaux annuels.
    Pour chaque mois t, le niveau utilise est celui de l'annee de ce mois.
    """
    T         = data["field"].shape[0]
    lo2       = data["lon2d"]; la2 = data["lat2d"]
    time      = data["time"]
    years_all = np.array([int(str(t)[:4]) for t in time])

    # Reference initiale = champ moyen, niveaux de reference
    mean_grad = compute_ssh_gradient(lo2, la2, data["mean_field"])
    clim_ref  = {}
    for fn in FRONT_NAMES:
        segs = extract_contour(lo2, la2, data["mean_field"], ref_levels[fn])
        seg, _ = select_main_branch_gradient(
            segs, mean_grad, lo2, la2, ref_levels[fn], ref_levels[fn])
        clim_ref[fn] = apply_lat_bounds(interp_to_lon_grid(seg), fn)

    phi  = {fn: np.full((T, len(LON_GRID)), np.nan) for fn in FRONT_NAMES}
    prev = {fn: clim_ref[fn].copy() for fn in FRONT_NAMES}

    print(f"  [Methode B - {prod_name}] extraction avec niveaux annuels, "
          f"{T} mois...", end="", flush=True)
    for t in range(T):
        if t % 24 == 0: print(f" {t}", end="", flush=True)
        y    = years_all[t]
        f2d  = data["field"][t]
        grad = compute_ssh_gradient(lo2, la2, f2d)
        ylevels = annual_levels.get(y, {fn: ref_levels[fn] for fn in FRONT_NAMES})

        for fn in FRONT_NAMES:
            lv    = ylevels[fn]
            segs  = extract_contour(lo2, la2, f2d, lv)
            seg, _ = select_main_branch_gradient(
                segs, grad, lo2, la2, ref_levels[fn], lv,
                clim_lat=clim_ref[fn], prev_lat=prev[fn])
            lat_t = apply_lat_bounds(interp_to_lon_grid(seg), fn)
            phi[fn][t] = lat_t
            prev[fn] = np.where(np.isfinite(lat_t), lat_t, prev[fn])
    print("  OK")
    return phi


# ─────────────────────────────────────────────
# ETAPE D — VALIDATION PAR GRADIENT SSH
# ─────────────────────────────────────────────

def find_gradient_core(phi_contour_t, grad2d, lon2d, lat2d,
                        band_deg=GRADIENT_SEARCH_DEG):
    """
    Pour chaque longitude, cherche le maximum de |nabla SSH| dans une
    fenetre +/- band_deg autour du contour.

    Retourne phi_grad_t (len(LON_GRID),) : latitude du max gradient.
    """
    phi_grad = np.full(len(LON_GRID), np.nan)
    for i, x in enumerate(LON_GRID):
        phi_c = phi_contour_t[i]
        if not np.isfinite(phi_c):
            continue
        msk = (np.abs(lon2d - x)   <= 1.0) & \
              (np.abs(lat2d - phi_c) <= band_deg)
        if not np.any(msk):
            continue
        lats_m = lat2d[msk]
        grad_m = grad2d[msk]
        ok     = np.isfinite(grad_m)
        if not np.any(ok):
            continue
        phi_grad[i] = lats_m[ok][np.argmax(grad_m[ok])]
    return phi_grad


def compute_gradient_validation(phi_fixed, phi_annual, data):
    """
    Etape D : validation par gradient SSH.

    Pour chaque mois et front, calcule :
      delta_fixed  = |phi_contour_fixe  - phi_gradient_max|  (deg)
      delta_annual = |phi_contour_annuel - phi_gradient_max|  (deg)

    Un delta faible signifie que le contour suit bien le coeur du jet.
    Un delta grand signifie que le contour s'est ecarte du maximum de gradient
    -> la detection est moins fiable.

    Retourne dict[fn] -> (delta_fixed, delta_annual) chacun (T, LON)
    """
    T   = data["field"].shape[0]
    lo2 = data["lon2d"]; la2 = data["lat2d"]
    res = {}

    print(f"  [Etape D] calcul gradient validation...", end="", flush=True)
    for fn in FRONT_NAMES:
        df = np.full((T, len(LON_GRID)), np.nan)
        da = np.full((T, len(LON_GRID)), np.nan)
        for t in range(T):
            if t % 48 == 0: print(f" {t}", end="", flush=True)
            grad = compute_ssh_gradient(lo2, la2, data["field"][t])
            # Dans compute_gradient_validation, pour chaque t :
            phi_g_for_fixed  = find_gradient_core(phi_fixed[fn][t],  grad, lo2, la2)
            phi_g_for_annual = find_gradient_core(phi_annual[fn][t], grad, lo2, la2)
            df[t] = np.abs(phi_fixed[fn][t]  - phi_g_for_fixed)
            da[t] = np.abs(phi_annual[fn][t] - phi_g_for_annual)
        res[fn] = {"delta_fixed": df, "delta_annual": da}
    print("  OK")
    return res


# ─────────────────────────────────────────────
# ETAPE E — VALIDATION PAR VITESSE GEOSTROPHIQUE (OBS)
# ─────────────────────────────────────────────

def find_speed_core(phi_contour_t, speed2d, lon2d, lat2d,
                    band_deg=GRADIENT_SEARCH_DEG):
    """
    Pour chaque longitude, cherche le maximum de vitesse geostrophique
    dans une fenetre +/- band_deg autour du contour.
    """
    phi_speed = np.full(len(LON_GRID), np.nan)
    for i, x in enumerate(LON_GRID):
        phi_c = phi_contour_t[i]
        if not np.isfinite(phi_c):
            continue
        msk = (np.abs(lon2d - x)   <= 1.0) & \
              (np.abs(lat2d - phi_c) <= band_deg)
        if not np.any(msk):
            continue
        lats_m  = lat2d[msk]
        speed_m = speed2d[msk]
        ok      = np.isfinite(speed_m)
        if not np.any(ok):
            continue
        phi_speed[i] = lats_m[ok][np.argmax(speed_m[ok])]
    return phi_speed


def compute_speed_validation(phi_fixed, phi_annual, data):
    """
    Etape E : validation par vitesse geostrophique (OBS uniquement).
    Meme logique que l'etape D mais avec |ug|^2 + |vg|^2 au lieu du gradient SSH.
    """
    if data["ug"] is None or data["vg"] is None:
        print("  [Etape E] vitesse geostrophique non disponible -> skip")
        return None

    T   = data["field"].shape[0]
    lo2 = data["lon2d"]; la2 = data["lat2d"]
    res = {}

    print(f"  [Etape E] calcul speed validation...", end="", flush=True)
    for fn in FRONT_NAMES:
        df = np.full((T, len(LON_GRID)), np.nan)
        da = np.full((T, len(LON_GRID)), np.nan)
        for t in range(T):
            if t % 48 == 0: print(f" {t}", end="", flush=True)
            speed = np.sqrt(data["ug"][t]**2 + data["vg"][t]**2)
            phi_s = find_speed_core(phi_fixed[fn][t], speed, lo2, la2)
            df[t] = np.abs(phi_fixed[fn][t]  - phi_s)
            da[t] = np.abs(phi_annual[fn][t] - phi_s)
        res[fn] = {"delta_fixed": df, "delta_annual": da}
    print("  OK")
    return res


# ─────────────────────────────────────────────
# ETAPE F — DESAISONNALISATION
# ─────────────────────────────────────────────

def remove_seasonal_cycle(phi_arrays, time):
    """
    Retire la climatologie mensuelle PAR LONGITUDE.
    Retourne dict[fn] -> array d'anomalies (T, LON).
    """
    months = np.array([int(str(t)[5:7]) for t in time])
    anom   = {}
    for fn in FRONT_NAMES:
        phi  = phi_arrays[fn]
        out  = np.full_like(phi, np.nan)
        clim = np.full((12, len(LON_GRID)), np.nan)
        for m in range(1, 13):
            sel = months == m
            if np.any(sel):
                clim[m-1] = np.nanmean(phi[sel], axis=0)
        for t, m in enumerate(months):
            out[t] = phi[t] - clim[m-1]
        anom[fn] = out
    return anom


# ─────────────────────────────────────────────
# DEPLACEMENT CIRCUMPOLAIRE
# ─────────────────────────────────────────────

def compute_displacement(phi_arrays, time, label_method=""):
    """
    Calcule le deplacement circumpolaire mensuel d_f(t) et les valeurs
    annuelles pour chaque front.
    """
    years_all  = np.array([int(str(t)[:4]) for t in time])
    unique_yrs = np.unique(years_all)
    results    = {}

    for fn in FRONT_NAMES:
        phi      = phi_arrays[fn]
        cov_mask = np.mean(np.isfinite(phi), axis=0) >= MIN_COVERAGE
        n_lon_ok = int(np.sum(cov_mask))
        phi_circ = np.nanmean(phi[:, cov_mask], axis=1)
        phi_mean = float(np.nanmean(phi_circ))
        d_km     = 111.2 * (phi_circ - phi_mean)

        phi_ann  = np.full(len(unique_yrs), np.nan)
        d_ann    = np.full(len(unique_yrs), np.nan)
        n_m_y    = np.zeros(len(unique_yrs), dtype=int)
        for iy, y in enumerate(unique_yrs):
            idx_y = years_all == y
            vals  = phi_circ[idx_y]
            valid = np.isfinite(vals)
            n_m_y[iy] = int(np.sum(valid))
            if np.sum(valid) >= 6:
                phi_ann[iy] = float(np.nanmean(vals[valid]))
                d_ann[iy]   = 111.2 * (phi_ann[iy] - phi_mean)

        results[fn] = dict(phi_circ=phi_circ, phi_mean=phi_mean,
                           d_km=d_km, phi_ann=phi_ann, d_ann_km=d_ann,
                           years=unique_yrs, n_months_y=n_m_y,
                           n_lon_ok=n_lon_ok)
    return results


# ─────────────────────────────────────────────
# TENDANCE + FILTRE
# ─────────────────────────────────────────────

def linear_trend_ar1(d_km):
    valid = np.isfinite(d_km)
    if np.sum(valid) < 12:
        return dict(slope_km_yr=np.nan, trend_line=np.full_like(d_km, np.nan),
                    pvalue=np.nan, r2=np.nan, total_km=np.nan, n_eff=0)
    x = np.arange(len(d_km))[valid].astype(float)
    y = d_km[valid]; N = len(y)
    slope, intercept, r, _, se = stats.linregress(x, y)
    resid = y - (slope * x + intercept)
    if N > 3:
        r1    = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0,1], -0.99, 0.99))
        n_eff = max(4, int(N * (1 - r1) / (1 + r1)))
    else:
        n_eff = N
    se_ar1 = se * np.sqrt(N / n_eff)
    pvalue = 2 * float(stats.t.sf(abs(slope / se_ar1), df=n_eff - 2))
    x_full = np.arange(len(d_km)).astype(float)
    trend_line = intercept + slope * x_full; trend_line[~valid] = np.nan
    N_years    = (np.sum(valid) - 1) / 12.0
    return dict(slope_km_yr=slope * 12.0, trend_line=trend_line,
                pvalue=pvalue, r2=r**2, total_km=slope * 12.0 * N_years,
                n_eff=n_eff)


def lowpass_12m(d_km, n=LOWPASS_MONTHS):
    T = len(d_km); out = np.full(T, np.nan); half = n // 2
    for t in range(half, T - half):
        window = d_km[t-half:t+half+1]; valid = np.isfinite(window)
        if np.sum(valid) >= n // 2:
            out[t] = np.nanmean(window[valid])
    return out


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def _fig8_core(disp_results, time, title_str, fname,
               color_bars=True):
    """Noyau de la figure 8 (factorise)."""
    T     = len(time)
    years = np.array([int(str(t)[:4]) + (int(str(t)[5:7])-0.5)/12. for t in time])
    t0 = str(time[0])[:7]; t1 = str(time[-1])[:7]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, fn in zip(axes, FRONT_NAMES):
        r    = disp_results[fn]
        d_km = r["d_km"]
        d_pos = np.where(d_km >= 0, d_km, 0.)
        d_neg = np.where(d_km <  0, d_km, 0.)
        bw    = 0.9 / 12.

        if color_bars:
            ax.bar(years, d_pos, width=bw, color="#d73027",
                   align="center", zorder=3, label="Equatorward")
            ax.bar(years, d_neg, width=bw, color="black",
                   align="center", zorder=3, label="Poleward")
        else:
            d_all = np.where(np.isfinite(d_km), d_km, 0.)
            ax.bar(years, d_all, width=bw, color="steelblue",
                   align="center", alpha=0.6, zorder=3, label="Deplacement")

        d_lp  = lowpass_12m(d_km)
        ok_lp = np.isfinite(d_lp)
        ax.plot(years[ok_lp], d_lp[ok_lp],
                color="#2166ac", lw=2.0, zorder=5, label=f"PB {LOWPASS_MONTHS}m")

        tr    = linear_trend_ar1(d_km)
        ok_tr = np.isfinite(tr["trend_line"])
        ax.plot(years[ok_tr], tr["trend_line"][ok_tr],
                color="#1a9641", lw=2.2, zorder=6, label="Tendance")

        ax.axhline(0, color="gray", lw=0.7, alpha=0.6, zorder=2)

        sig = "**" if tr["pvalue"] < 0.05 else ("*" if tr["pvalue"] < 0.10 else "n.s.")
        ax.text(0.02, 0.97,
                f"{tr['slope_km_yr']:.1f} km/an  "
                f"({tr['total_km']:.0f} km)  p={tr['pvalue']:.3f}{sig}  "
                f"n_eff={tr.get('n_eff','?')}",
                transform=ax.transAxes, fontsize=8, va="top", color="#1a9641",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1a9641", alpha=0.8))
        ax.text(0.98, 0.97,
                f"phi_mean={r['phi_mean']:.1f}°S  N_lon={r['n_lon_ok']}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", lw=0.3, alpha=0.7, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=9)
        ax.set_ylabel("Deplacement (km)", fontsize=10)
        ylim = max(70, np.nanmax(np.abs(d_km)) * 1.15 if np.any(np.isfinite(d_km)) else 70)
        ax.set_ylim(-ylim, ylim)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles, lbls, fontsize=7.5, loc="lower right",
                  framealpha=0.9, edgecolor="#cccccc", ncol=3)
        ax.set_title(f"({chr(97+FRONT_NAMES.index(fn))}) {FRONT_LABELS[fn]}",
                     fontsize=10, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Annee", fontsize=11)
    axes[-1].set_xlim(years[0]-0.1, years[-1]+0.1)
    fig.suptitle(title_str, fontsize=9.5, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[FIG]  {out}")


def fig_bias(phi_fixed, phi_annual, time, prod_name):
    """
    Etape C : biais fixe - annuel = phi_fixed - phi_annual.
    Si positif : la methode fixe place le front plus au nord.
    Si negatif : la methode fixe place le front plus au sud.
    Un biais croissant dans le temps signale une tendance artefactuelle
    due a la derive du niveau de mer large echelle.
    """
    years = np.array([int(str(t)[:4]) + (int(str(t)[5:7])-0.5)/12. for t in time])
    t0 = str(time[0])[:7]; t1 = str(time[-1])[:7]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, fn in zip(axes, FRONT_NAMES):
        bias_t = 111.2 * (phi_fixed[fn] - phi_annual[fn])
        bias_c = np.nanmean(bias_t, axis=1)   # moyenne circumpolaire

        ok = np.isfinite(bias_c)
        ax.fill_between(years[ok],  bias_c[ok], 0,
                        where=bias_c[ok] >= 0, color="#d73027",
                        alpha=0.45, label="Fixe > Annuel (nord)")
        ax.fill_between(years[ok],  bias_c[ok], 0,
                        where=bias_c[ok] < 0, color="#4393c3",
                        alpha=0.45, label="Fixe < Annuel (sud)")
        ax.plot(years[ok], bias_c[ok], color="black", lw=0.8, alpha=0.5)

        lp = lowpass_12m(bias_c)
        ok_lp = np.isfinite(lp)
        ax.plot(years[ok_lp], lp[ok_lp],
                color="black", lw=2.0, label=f"PB {LOWPASS_MONTHS}m")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)

        tr  = linear_trend_ar1(bias_c)
        ok_tr = np.isfinite(tr["trend_line"])
        ax.plot(years[ok_tr], tr["trend_line"][ok_tr],
                color="#e31a1c", lw=1.8, ls="--", label="Tendance du biais")

        sig = "**" if tr["pvalue"] < 0.05 else ("*" if tr["pvalue"] < 0.10 else "n.s.")
        ax.text(0.02, 0.97,
                f"Tendance biais : {tr['slope_km_yr']:.2f} km/an  "
                f"p={tr['pvalue']:.3f}{sig}",
                transform=ax.transAxes, fontsize=8, va="top",
                color="#e31a1c",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#e31a1c", alpha=0.8))

        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", lw=0.3, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=9)
        ax.set_ylabel("Biais (km)", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right", ncol=3,
                  framealpha=0.9, edgecolor="#cccccc")
        ax.set_title(f"({chr(97+FRONT_NAMES.index(fn))}) {FRONT_LABELS[fn]}",
                     fontsize=10, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Annee", fontsize=11)
    axes[-1].set_xlim(years[0]-0.1, years[-1]+0.1)
    fig.suptitle(
        f"Biais de position : methode fixe - methode annuelle — {prod_name}  "
        f"({t0} -> {t1})\n"
        f"111.2 * (phi_fixe - phi_annuel)  km  "
        f"[rouge = fixe trop au nord / bleu = fixe trop au sud]\n"
        f"Tendance du biais = derive artefactuelle potentielle de la methode fixe",
        fontsize=9.5, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"fig_bias_{prod_name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[FIG]  {out}")


def fig_gradient_validation(grad_val, time, prod_name):
    """
    Etape D : distance contour / maximum de gradient SSH.
    Montre si le contour SSH suit bien le coeur du jet.
    delta faible = bonne correspondance ; delta grand = contour ecarte du jet.
    """
    years = np.array([int(str(t)[:4]) + (int(str(t)[5:7])-0.5)/12. for t in time])
    t0 = str(time[0])[:7]; t1 = str(time[-1])[:7]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, fn in zip(axes, FRONT_NAMES):
        df_circ = np.nanmean(grad_val[fn]["delta_fixed"],  axis=1)
        da_circ = np.nanmean(grad_val[fn]["delta_annual"], axis=1)

        ok_f = np.isfinite(df_circ); ok_a = np.isfinite(da_circ)
        ax.plot(years[ok_f], df_circ[ok_f],
                color=FRONT_COLORS[fn], lw=1.2, alpha=0.5,
                label="Methode fixe (brut)")
        ax.plot(years[ok_a], da_circ[ok_a],
                color="black", lw=1.2, alpha=0.5, ls="--",
                label="Methode annuelle (brut)")

        lp_f = lowpass_12m(df_circ); lp_a = lowpass_12m(da_circ)
        ok_lf = np.isfinite(lp_f); ok_la = np.isfinite(lp_a)
        ax.plot(years[ok_lf], lp_f[ok_lf],
                color=FRONT_COLORS[fn], lw=2.2,
                label=f"Fixe lisse")
        ax.plot(years[ok_la], lp_a[ok_la],
                color="black", lw=2.2, ls="--",
                label=f"Annuel lisse")

        mean_f = np.nanmean(df_circ); mean_a = np.nanmean(da_circ)
        ax.text(0.02, 0.97,
                f"<delta_fixe> = {mean_f:.2f} deg  |  "
                f"<delta_annuel> = {mean_a:.2f} deg  |  "
                f"En km : fixe = {mean_f*111.2:.0f}  annuel = {mean_a*111.2:.0f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", lw=0.3, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=9)
        ax.set_ylabel("|phi_contour - phi_grad| (deg)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2,
                  framealpha=0.9, edgecolor="#cccccc")
        ax.set_title(f"({chr(97+FRONT_NAMES.index(fn))}) {FRONT_LABELS[fn]}",
                     fontsize=10, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Annee", fontsize=11)
    axes[-1].set_xlim(years[0]-0.1, years[-1]+0.1)
    fig.suptitle(
        f"Distance contour SSH / maximum de gradient SSH — {prod_name}  "
        f"({t0} -> {t1})\n"
        f"delta = |phi_contour - phi_grad_max|  deg  |  "
        f"Delta faible = contour bien aligne sur le jet\n"
        f"[Validation methodologique : etape D]",
        fontsize=9.5, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"fig_gradient_val_{prod_name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[FIG]  {out}")


def fig_speed_validation(speed_val, time, prod_name):
    """Etape E : distance contour / maximum de vitesse geostrophique (OBS)."""
    if speed_val is None:
        return
    years = np.array([int(str(t)[:4]) + (int(str(t)[5:7])-0.5)/12. for t in time])
    t0 = str(time[0])[:7]; t1 = str(time[-1])[:7]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, fn in zip(axes, FRONT_NAMES):
        df_c = np.nanmean(speed_val[fn]["delta_fixed"],  axis=1)
        da_c = np.nanmean(speed_val[fn]["delta_annual"], axis=1)
        ok_f = np.isfinite(df_c); ok_a = np.isfinite(da_c)
        ax.plot(years[ok_f], df_c[ok_f], color=FRONT_COLORS[fn],
                lw=1.2, alpha=0.5, label="Fixe (brut)")
        ax.plot(years[ok_a], da_c[ok_a], color="black",
                lw=1.2, alpha=0.5, ls="--", label="Annuel (brut)")
        lp_f = lowpass_12m(df_c); lp_a = lowpass_12m(da_c)
        ok_lf = np.isfinite(lp_f); ok_la = np.isfinite(lp_a)
        ax.plot(years[ok_lf], lp_f[ok_lf], color=FRONT_COLORS[fn],
                lw=2.2, label="Fixe lisse")
        ax.plot(years[ok_la], lp_a[ok_la], color="black",
                lw=2.2, ls="--", label="Annuel lisse")
        mf = np.nanmean(df_c); ma_ = np.nanmean(da_c)
        ax.text(0.02, 0.97,
                f"<delta_fixe> = {mf:.2f} deg ({mf*111.2:.0f} km)  |  "
                f"<delta_annuel> = {ma_:.2f} deg ({ma_*111.2:.0f} km)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", lw=0.3, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=9)
        ax.set_ylabel("|phi_contour - phi_speed| (deg)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2,
                  framealpha=0.9, edgecolor="#cccccc")
        ax.set_title(f"({chr(97+FRONT_NAMES.index(fn))}) {FRONT_LABELS[fn]}",
                     fontsize=10, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Annee", fontsize=11)
    axes[-1].set_xlim(years[0]-0.1, years[-1]+0.1)
    fig.suptitle(
        f"Distance contour SSH / maximum de vitesse geostrophique — {prod_name}  "
        f"({t0} -> {t1})\n"
        f"delta = |phi_contour - phi_speed_max|  deg  |  "
        f"Delta faible = contour sur le jet\n"
        f"[Validation methodologique : etape E]",
        fontsize=9.5, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"fig_speed_val_{prod_name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[FIG]  {out}")


# ─────────────────────────────────────────────
# EXPORTS CSV
# ─────────────────────────────────────────────

def save_annual_levels_csv(level_records, prod_name):
    path = os.path.join(OUTDIR, f"annual_levels_{prod_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "year","front","level_fixed","level_annual",
            "delta_level","n_months","score","note"])
        w.writeheader(); w.writerows(level_records)
    print(f"[CSV]  {path}")


def save_annual_table_csv(disp_fixed, disp_annual, time, prod_name):
    years_all  = np.array([int(str(t)[:4]) for t in time])
    unique_yrs = np.unique(years_all)
    path = os.path.join(OUTDIR, f"table_annual_{prod_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "annee",
            "phi_SAF_fixed","d_SAF_fixed_km",
            "phi_PF_fixed","d_PF_fixed_km",
            "phi_SACCF_fixed","d_SACCF_fixed_km",
            "phi_SAF_annual","d_SAF_annual_km",
            "phi_PF_annual","d_PF_annual_km",
            "phi_SACCF_annual","d_SACCF_annual_km",
            "N_mois_min",
        ])
        # Ligne de reference
        refs_f = [disp_fixed[fn]["phi_mean"]  for fn in FRONT_NAMES]
        refs_a = [disp_annual[fn]["phi_mean"] for fn in FRONT_NAMES]
        w.writerow(["phi_mean_fixe",
                    *[f"{v:.3f}" for v in refs_f], "0.0",
                    *["" for _ in range(6)]])
        w.writerow(["phi_mean_annuel",
                    *["" for _ in range(6)],
                    *[f"{v:.3f}" for v in refs_a], "0.0"])
        w.writerow([])
        for y in unique_yrs:
            row_data = [str(y)]
            n_list   = []
            for disp in [disp_fixed, disp_annual]:
                for fn in FRONT_NAMES:
                    r   = disp[fn]
                    yrs = list(r["years"])
                    if y in yrs:
                        iy = yrs.index(y)
                        phi = r["phi_ann"][iy]
                        d   = r["d_ann_km"][iy]
                        n_list.append(int(r["n_months_y"][iy]))
                        fmt_phi = f"{phi:.2f}" if np.isfinite(phi) else "NaN"
                        fmt_d   = f"{d:.1f}"  if np.isfinite(d) else "NaN"
                    else:
                        fmt_phi, fmt_d = "NaN", "NaN"
                    row_data += [fmt_phi, fmt_d]
            n_min = min(n_list) if n_list else 0
            row_data.append(str(n_min) + ("*" if n_min < 12 else ""))
            w.writerow(row_data)
    print(f"[CSV]  {path}")


# ─────────────────────────────────────────────
# RAPPORT TERMINAL
# ─────────────────────────────────────────────

def print_summary(prod_name, time, disp_fixed, disp_annual):
    t0 = str(time[0])[:7]; t1 = str(time[-1])[:7]
    print(f"\n{'='*78}")
    print(f"  RESUME — {prod_name}  ({t0} -> {t1})")
    print(f"{'='*78}")
    print(f"  {'Front':8s}  {'Methode':8s}  {'phi_mean':>10s}  "
          f"{'slope':>10s}  {'total':>10s}  {'p-val':>8s}  {'sig':>5s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    for fn in FRONT_NAMES:
        for label, disp in [("fixe", disp_fixed), ("annuel", disp_annual)]:
            r  = disp[fn]
            tr = linear_trend_ar1(r["d_km"])
            sig = "**" if tr["pvalue"]<0.05 else ("*" if tr["pvalue"]<0.10 else "n.s.")
            print(f"  {fn:8s}  {label:8s}  "
                  f"{r['phi_mean']:>10.2f} dS  "
                  f"{tr['slope_km_yr']:>8.2f} km/an  "
                  f"{tr['total_km']:>8.1f} km  "
                  f"{tr['pvalue']:>8.4f}  {sig:>5s}")
    print(f"{'='*78}\n")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run(data, prod_name):
    ref_lv = REF_LEVELS[prod_name]
    time   = data["time"]

    print(f"\n{'═'*68}")
    print(f"  {prod_name}  ({str(time[0])[:7]} -> {str(time[-1])[:7]})")
    print(f"{'═'*68}")

    # ── Etape A : methode fixe ───────────────────────────────────────
    print("\n[Etape A] Methode fixe (Kim & Orsi 2014)...")
    phi_fixed = extract_phi_fixed(data, ref_lv, prod_name)
    disp_fixed = compute_displacement(phi_fixed, time)
    _fig8_core(disp_fixed, time,
               (f"Methode FIXE (Kim & Orsi 2014) — {prod_name}  "
                f"({str(time[0])[:7]} -> {str(time[-1])[:7]})\n"
                f"Niveaux SSH constants : SAF={ref_lv['SAF']}m  "
                f"PF={ref_lv['PF']}m  SACCF={ref_lv['SACCF']}m\n"
                f"Rouge=equatorward | Noir=poleward | "
                f"Bleu=PB {LOWPASS_MONTHS}m | Vert=tendance (AR1 corr.)"),
               f"fig8_fixed_{prod_name}.png")

    # ── Etape B : niveaux annuels ────────────────────────────────────
    print("\n[Etape B] Recalibrage annuel des niveaux SSH...")
    annual_levels, level_records = estimate_annual_front_levels(
        data, ref_lv, prod_name)
    save_annual_levels_csv(level_records, prod_name)

    phi_annual = extract_phi_annual(data, annual_levels, ref_lv, prod_name)
    disp_annual = compute_displacement(phi_annual, time)
    _fig8_core(disp_annual, time,
               (f"Methode ANNUELLE — {prod_name}  "
                f"({str(time[0])[:7]} -> {str(time[-1])[:7]})\n"
                f"Niveaux SSH recalibres chaque annee "
                f"(ref +/- {SEARCH_LEVEL_RANGE}m, pas {SEARCH_LEVEL_STEP}m)\n"
                f"Rouge=equatorward | Noir=poleward | "
                f"Bleu=PB {LOWPASS_MONTHS}m | Vert=tendance (AR1 corr.)"),
               f"fig8_annual_{prod_name}.png")

    # ── Etape C : biais fixe - annuel ───────────────────────────────
    print("\n[Etape C] Biais fixe vs annuel...")
    fig_bias(phi_fixed, phi_annual, time, prod_name)

    # ── Etape F : desaisonnalisation ────────────────────────────────
    print("\n[Etape F] Desaisonnalisation...")
    anom_fixed  = remove_seasonal_cycle(phi_fixed,  time)
    anom_annual = remove_seasonal_cycle(phi_annual, time)
    disp_fixed_deseas  = compute_displacement(
        {fn: phi_fixed[fn]  + anom_fixed[fn]  - phi_fixed[fn]
         for fn in FRONT_NAMES}, time)
    # Plus directement : calculer le deplacement sur les anomalies
    # en conservant phi_mean de la methode brute comme reference
    # -> phi_circ_anom(t) = nanmean(phi_anom(t)) -> d = 111.2 * phi_circ_anom
    def disp_from_anom(anom, phi_mean_dict):
        cov_mask = {fn: np.mean(np.isfinite(anom[fn]), axis=0) >= MIN_COVERAGE
                    for fn in FRONT_NAMES}
        res = {}
        years_all = np.array([int(str(t)[:4]) for t in time])
        unique_yrs = np.unique(years_all)
        for fn in FRONT_NAMES:
            circ = np.nanmean(anom[fn][:, cov_mask[fn]], axis=1)
            d_km = 111.2 * circ   # anomalies : phi_mean = 0 par construction
            phi_ann = np.full(len(unique_yrs), np.nan)
            d_ann   = np.full(len(unique_yrs), np.nan)
            n_m_y   = np.zeros(len(unique_yrs), dtype=int)
            for iy, y in enumerate(unique_yrs):
                idx_y = years_all == y; vals = circ[idx_y]
                valid = np.isfinite(vals); n_m_y[iy] = int(np.sum(valid))
                if np.sum(valid) >= 6:
                    phi_ann[iy] = float(np.nanmean(vals[valid]))
                    d_ann[iy]   = 111.2 * phi_ann[iy]
            res[fn] = dict(phi_circ=circ, phi_mean=0.0, d_km=d_km,
                           phi_ann=phi_ann, d_ann_km=d_ann,
                           years=unique_yrs, n_months_y=n_m_y,
                           n_lon_ok=int(np.sum(cov_mask[fn])))
        return res

    disp_fd = disp_from_anom(anom_fixed,  {fn: disp_fixed[fn]["phi_mean"]
                                            for fn in FRONT_NAMES})
    disp_ad = disp_from_anom(anom_annual, {fn: disp_annual[fn]["phi_mean"]
                                            for fn in FRONT_NAMES})

    _fig8_core(disp_fd, time,
               (f"Methode FIXE desaisonnalisee — {prod_name}  "
                f"({str(time[0])[:7]} -> {str(time[-1])[:7]})\n"
                f"phi'_circ(t) = anomalie circumpolaire apres retrait "
                f"de la climatologie mensuelle\n"
                f"Tendance = derive decennale (hors cycle saisonnier)"),
               f"fig8_fixed_deseas_{prod_name}.png", color_bars=False)

    _fig8_core(disp_ad, time,
               (f"Methode ANNUELLE desaisonnalisee — {prod_name}  "
                f"({str(time[0])[:7]} -> {str(time[-1])[:7]})\n"
                f"phi'_circ(t) = anomalie circumpolaire (niveaux annuels + "
                f"retrait climatologie mensuelle)"),
               f"fig8_annual_deseas_{prod_name}.png", color_bars=False)

    # ── Etape D : validation gradient ───────────────────────────────
    print("\n[Etape D] Validation par gradient SSH...")
    grad_val = compute_gradient_validation(phi_fixed, phi_annual, data)
    fig_gradient_validation(grad_val, time, prod_name)

    # ── Etape E : validation vitesse (OBS uniquement) ───────────────
    print("\n[Etape E] Validation par vitesse geostrophique...")
    speed_val = compute_speed_validation(phi_fixed, phi_annual, data)
    fig_speed_validation(speed_val, time, prod_name)

    # ── Export CSV tableau annuel ────────────────────────────────────
    save_annual_table_csv(disp_fixed, disp_annual, time, prod_name)

    # ── Resume terminal ──────────────────────────────────────────────
    print_summary(prod_name, time, disp_fixed, disp_annual)

    return dict(phi_fixed=phi_fixed, phi_annual=phi_annual,
                disp_fixed=disp_fixed, disp_annual=disp_annual,
                grad_val=grad_val, speed_val=speed_val)


def main():
    print("\n" + "=" * 68)
    print("  ACC_FRONT_METHODS.py")
    print("  Pipeline methodologique complet (etapes A a F)")
    print("=" * 68)
    print(f"  Sorties -> {OUTDIR}/\n")

    try:
        obs = load_obs(OBS_FILE, OBS_T0, OBS_T1)
        run(obs, "OBS")
    except FileNotFoundError as e:
        print(f"[SKIP OBS] {e}")

    try:
        nemo = load_nemo(MOD_FILE, MOD_T0, MOD_T1)
        run(nemo, "NEMO")
    except FileNotFoundError as e:
        print(f"[SKIP NEMO] {e}")

    print(f"\n[DONE]  Sorties dans : {OUTDIR}/")
    print("\nFichiers produits :")
    for f in [
        "fig8_fixed_<PROD>.png         — Etape A : methode fixe (Kim & Orsi)",
        "fig8_annual_<PROD>.png        — Etape B : niveaux SSH annuels",
        "fig_bias_<PROD>.png           — Etape C : biais fixe - annuel",
        "fig8_fixed_deseas_<PROD>.png  — Etape F : fixe + desaisonnalise",
        "fig8_annual_deseas_<PROD>.png — Etape F : annuel + desaisonnalise",
        "fig_gradient_val_<PROD>.png   — Etape D : distance contour / grad SSH",
        "fig_speed_val_OBS.png         — Etape E : distance contour / vitesse",
        "annual_levels_<PROD>.csv      — Niveaux SSH annuels retenus",
        "table_annual_<PROD>.csv       — Positions et deplacements annuels",
    ]:
        print(f"  {f}")


if __name__ == "__main__":
    main()