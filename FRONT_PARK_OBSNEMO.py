"""
FRONT_PARK_OBSNEMO.py
=====================
Identification des fronts de l'ACC selon Park et al. (2019), JGR Oceans.

Pipeline fidèle à Park :
  1. Champ moyen DOT/ZOS + vitesse
  2. Extraction de TOUS les contours candidats
  3. Filtrage "contour circumpolaire"
  4. NB/SB définis à Drake Passage (porte géographique)
  5. Boîtes choke points (UFZ, DP, SWIR, KP, TAS)
  6. Scoring des contours candidats entre NB et SB
  7. SAF, PF, SACCF = contours au meilleur score
  8. Largeurs + choke points + figures

Usage : python FRONT_PARK_OBSNEMO.py
"""

# ================================================================
# BLOC 0 — imports
# ================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ================================================================
# BLOC 1 — CONFIGURATION
# ================================================================

OBS_FILE   = "dot_all_30bmedian_eigen6s4v2_sig3.nc"
MOD_FILE   = "zos_1991_2023_IS.nc"
OUTDIR     = "./outputs_park_v2"

LAT_MIN, LAT_MAX = -80, -30

# Grille commune
LON_GRID = np.arange(-180, 181, 1.0)

# Résolution du scan de niveaux candidats (m)
LEVEL_STEP = 0.02        # Park travaille au 0.1 m, on scanne plus fin

# Seuil "contour quasi-circumpolaire"
MIN_LON_COVERAGE = 270   # degrés de longitude couverts minimum

# ---- Porte Drake Passage (Park : limites continentales du passage) ----
DP_GATE = {
    "lon_min": -68, "lon_max": -54,
    "lat_min": -65, "lat_max": -54,
}

# ---- 5 grandes boîtes choke points (Park Fig. 4/5) ----
CHOKE_BOXES = {
    "UFZ":  {"lon_min": -150, "lon_max": -138, "lat_min": -60, "lat_max": -51},
    "DP":   {"lon_min": -68,  "lon_max": -54,  "lat_min": -65, "lat_max": -54},
    "SWIR": {"lon_min": 20,   "lon_max": 40,   "lat_min": -56, "lat_max": -44},
    "KP":   {"lon_min": 64,   "lon_max": 84,   "lat_min": -58, "lat_max": -46},
    "TAS":  {"lon_min": 140,  "lon_max": 165,  "lat_min": -58, "lat_max": -45},
}

# Poids des boîtes dans le score (tous égaux par défaut)
CHOKE_WEIGHTS = {k: 1.0 for k in CHOKE_BOXES}

# ---- Contraintes de choke points par produit ----
# OBS  : champ limité à ~50°S → SWIR (~46°S) et KP (~50°S) souvent hors champ.
#        On exige que le contour touche au moins ces 3 boîtes accessibles.
# NEMO : pas de restriction, on utilise toutes les 5 boîtes.
CHOKE_MIN_HIT = {
    "OBS":  {"required": ["UFZ", "DP", "TAS"], "min_count": 3},
    "MOD":  {"required": [],                   "min_count": 1},
}

# Seuil vitesse jet Park (m/s)
JET_THRESHOLD = 0.20

# Bande de recherche pour interpolation fronts → lon_grid
SEARCH_BAND_DEG = 1.5

# Minimum de points pour qu'un segment soit valide
MIN_SEGMENT_PTS = 30

# Ordres méridiens attendus (Park) : SAF > PF > SACCF (latitudes moins négatives)
# → niveaux DOT décroissants du nord au sud chez Park : NB > SAF > PF > SACCF > SB

# Styles graphiques
FRONT_STYLE = {
    "NB":    {"color": "magenta", "lw": 2.0, "ls": "-",  "label": "NB"},
    "SAF":   {"color": "black",   "lw": 2.2, "ls": "-",  "label": "SAF"},
    "PF":    {"color": "black",   "lw": 2.2, "ls": "--", "label": "PF"},
    "SACCF": {"color": "black",   "lw": 2.2, "ls": ":",  "label": "SACCF"},
    "SB":    {"color": "magenta", "lw": 2.0, "ls": "-",  "label": "SB"},
}

KNOWN_CHOKE_LON = {"UFZ": -144, "DP": -62, "SWIR": 30, "KP": 72, "TAS": 147}

# ================================================================
# BLOC 2 — CHARGEMENT
# ================================================================

def remap_curvilinear(lon, lat, field):
    nx, ny = lon.shape
    nlo = np.zeros_like(lon)
    nla = np.zeros_like(lat)
    nfi = np.full_like(field, np.nan, dtype=float)
    for i in range(nx):
        j = int(np.argmin(lon[i, :]))
        nlo[i, :] = np.roll(lon[i, :],   -j)
        nla[i, :] = np.roll(lat[i, :],   -j)
        nfi[i, :] = np.roll(field[i, :], -j)
    nlo = np.where(nlo > 180, nlo - 360, nlo)
    return nlo, nla, nfi


def load_obs(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    ds = xr.open_dataset(fpath)
    dot = ds["dot"]
    fm  = dot.mean("time") if "time" in dot.dims else dot
    if "land_mask" in ds:
        fm = fm.where(ds["land_mask"] == 0)
    if fm.dims[0] != "latitude":
        fm = fm.transpose("latitude", "longitude")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon = np.where(lon > 180, lon - 360, lon)
    idx = np.argsort(lon); lon = lon[idx]
    fv  = fm.values[:, idx].astype(float)
    LO, LA = np.meshgrid(lon, lat)

    sp = None
    if "ug" in ds and "vg" in ds:
        ug = ds["ug"].mean("time") if "time" in ds["ug"].dims else ds["ug"]
        vg = ds["vg"].mean("time") if "time" in ds["vg"].dims else ds["vg"]
        if ug.dims[0] != "latitude":
            ug = ug.transpose("latitude", "longitude")
            vg = vg.transpose("latitude", "longitude")
        sp = np.sqrt(ug.values[:, idx]**2 + vg.values[:, idx]**2)

    print(f"[OBS] shape={fv.shape}  lon=[{lon.min():.1f},{lon.max():.1f}]"
          f"  DOT=[{np.nanmin(fv):.2f},{np.nanmax(fv):.2f}] m")
    return dict(name="OBS", lon2d=LO, lat2d=LA, lon1d=lon, lat1d=lat,
                field=fv, speed=sp)


def load_mod(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    ds  = xr.open_dataset(fpath)
    zos = ds["zos"]
    zm  = zos.mean("time_counter") if "time_counter" in zos.dims else zos
    zm  = zm.where(zm != 0)
    lo2, la2, fv = remap_curvilinear(
        ds["nav_lon"].values, ds["nav_lat"].values, zm.values.astype(float))
    print(f"[MOD] shape={fv.shape}  ZOS=[{np.nanmin(fv):.2f},{np.nanmax(fv):.2f}] m")
    return dict(name="MOD", lon2d=lo2, lat2d=la2, lon1d=None, lat1d=None,
                field=fv, speed=None)


def detect_data_extent(d, nan_frac_thresh=0.90):
    """
    Détecte les latitudes extrêmes où le champ est effectivement non-NaN
    sur au moins (1-nan_frac_thresh) des points de la ligne.
    Nécessaire quand le produit OBS ne couvre pas toute la bande voulue.
    """
    f     = d["field"]
    lat2  = d["lat2d"]
    lats  = np.unique(np.round(lat2.ravel(), 2))
    valid = []
    for la in lats:
        row = f[np.abs(lat2 - la) < 0.6]
        if len(row) == 0:
            continue
        if np.sum(np.isnan(row)) / len(row) < nan_frac_thresh:
            valid.append(la)
    if not valid:
        return LAT_MIN, LAT_MAX
    ls, ln = float(np.min(valid)), float(np.max(valid))
    print(f"  [extent] données valides : lat=[{ls:.1f}, {ln:.1f}]")
    return ls, ln


def mask_latitudes(d, lat_min=None, lat_max=None):
    """
    Masque hors bande latitudinale.
    Détecte automatiquement l'étendue réelle si lat_min/lat_max non fournis.
    Stocke lat_min_eff / lat_max_eff dans le dict pour usage aval.
    """
    if lat_min is None or lat_max is None:
        ls, ln = detect_data_extent(d)
        lat_min = max(LAT_MIN, ls)
        lat_max = min(LAT_MAX, ln)

    m   = (d["lat2d"] < lat_min) | (d["lat2d"] > lat_max)
    out = d.copy()
    out["lat_min_eff"] = lat_min
    out["lat_max_eff"] = lat_max

    f = d["field"].copy(); f[m] = np.nan; out["field"] = f
    if d["speed"] is not None:
        s = d["speed"].copy(); s[m] = np.nan; out["speed"] = s
    return out


# ================================================================
# BLOC 3 — EXTRACTION DES CONTOURS
# ================================================================

def _extract_paths_from_cs(cs):
    """Compatibilité matplotlib < 3.8 et >= 3.8."""
    try:
        return cs.get_paths()           # matplotlib >= 3.8
    except AttributeError:
        if cs.collections:
            return cs.collections[0].get_paths()
        return []


def clean_segment_boundary(seg, d, border_margin_deg=0.1):
    """
    Supprime les artefacts de bord dans les segments de contour.

    Deux filtres :
      1. Retrait des points à moins de border_margin_deg du bord latitudinal
         du champ (très petit : 0.1° pour ne pas amputer NB/SB qui sont
         légitimement proches du bord du champ).
      2. Retrait des séquences "plates" : latitude quasi constante sur une
         fenêtre glissante → signature d'un contour qui longe un masque
         continent ou le bord de la grille.
    """
    lo, la = seg["lon"].copy(), seg["lat"].copy()

    # --- filtre 1 : bord latitudinal (marge intentionnellement faible) ---
    lat_s = d.get("lat_min_eff", LAT_MIN)
    lat_n = d.get("lat_max_eff", LAT_MAX)
    ok = (la > lat_s + border_margin_deg) & (la < lat_n - border_margin_deg)
    lo, la = lo[ok], la[ok]
    if len(lo) < MIN_SEGMENT_PTS:
        return None

    # --- filtre 2 : séquences plates (artefact masque continent) ---
    # On ne retire que si la variance locale est quasi nulle ET si on est
    # à moins de 2° d'un bord (pour ne pas couper des fronts zonaux larges).
    window = 5
    keep = np.ones(len(lo), dtype=bool)
    for i in range(window, len(lo) - window):
        near_south = la[i] < lat_s + 2.0
        near_north = la[i] > lat_n - 2.0
        if near_south or near_north:
            local_range = (np.max(la[i-window:i+window])
                           - np.min(la[i-window:i+window]))
            if local_range < 0.05:
                keep[i] = False
    lo, la = lo[keep], la[keep]
    if len(lo) < MIN_SEGMENT_PTS:
        return None

    return {"lon": lo, "lat": la, "level": seg["level"]}


def extract_contour_segments(lon2d, lat2d, field, level):
    """Extrait tous les segments du contour `level`."""
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(lon2d, lat2d, field, levels=[level])
    except Exception:
        plt.close(fig); return []
    plt.close(fig)

    segs = []
    for path in _extract_paths_from_cs(cs):
        v  = path.vertices
        lo, la = v[:, 0], v[:, 1]
        ok = np.isfinite(lo) & np.isfinite(la)
        lo, la = lo[ok], la[ok]
        if len(lo) >= MIN_SEGMENT_PTS:
            segs.append({"lon": lo, "lat": la, "level": level})
    return segs


def is_circumpolar(seg, min_coverage=MIN_LON_COVERAGE):
    """Vrai si le segment couvre suffisamment de longitudes."""
    lo = seg["lon"]
    return (np.nanmax(lo) - np.nanmin(lo)) >= min_coverage


def choose_main_segment(segs):
    """Segment avec la plus grande couverture zonale."""
    if not segs:
        return None
    return max(segs, key=lambda s: np.nanmax(s["lon"]) - np.nanmin(s["lon"]))


# ================================================================
# BLOC 4 — SCAN DE TOUS LES NIVEAUX CANDIDATS
# ================================================================

def scan_all_levels(d, step=LEVEL_STEP):
    """
    Pour chaque niveau, extrait les segments circumpolaires nettoyés.
    Renvoie une liste de {'level': float, 'seg': dict}.
    """
    field = d["field"]
    lo2, la2 = d["lon2d"], d["lat2d"]

    lo_all = np.arange(
        np.floor(np.nanmin(field) / step) * step,
        np.ceil( np.nanmax(field) / step) * step + step,
        step
    )

    results = []
    print(f"  Scan de {len(lo_all)} niveaux (step={step} m)…", end="", flush=True)
    for lv in lo_all:
        segs = extract_contour_segments(lo2, la2, field, lv)
        # nettoyage bord continent / bord de champ
        segs_clean = []
        for s in segs:
            sc = clean_segment_boundary(s, d)
            if sc is not None:
                segs_clean.append(sc)
        circ = [s for s in segs_clean if is_circumpolar(s)]
        main = choose_main_segment(circ)
        if main is not None:
            results.append({"level": lv, "seg": main})
    print(f"  → {len(results)} niveaux avec contour complet propre")
    return results


def adapt_dp_gate(d):
    """
    Adapte la porte Drake Passage à l'étendue latitudinale effective
    des données. Si le champ ne descend pas jusqu'à -65°S, on remonte
    la limite sud de la porte en conséquence.
    """
    lat_s = d.get("lat_min_eff", LAT_MIN)
    lat_n = d.get("lat_max_eff", LAT_MAX)

    gate = DP_GATE.copy()
    # Borne sud : ne pas descendre sous lat_min_eff + marge
    gate["lat_min"] = max(DP_GATE["lat_min"], lat_s + 0.5)
    # Borne nord : ne pas dépasser lat_max_eff - marge
    gate["lat_max"] = min(DP_GATE["lat_max"], lat_n - 0.5)

    if gate["lat_min"] >= gate["lat_max"]:
        print(f"  [WARN] porte Drake invalide après adaptation "
              f"({gate['lat_min']:.1f} – {gate['lat_max']:.1f}) "
              f"— champ trop court en latitude ?")
        # Fallback : prendre toute la plage disponible autour de DP
        gate["lat_min"] = lat_s + 0.5
        gate["lat_max"] = lat_n - 0.5

    print(f"  [Drake gate] lon=[{gate['lon_min']}, {gate['lon_max']}]  "
          f"lat=[{gate['lat_min']:.1f}, {gate['lat_max']:.1f}]")
    return gate


# ================================================================
# BLOC 5 — NB / SB À DRAKE PASSAGE
# ================================================================

def lat_in_gate(seg, gate):
    """
    Latitudes du segment dans la porte géographique.
    Renvoie tableau vide si le segment n'y passe pas.
    """
    lo, la = seg["lon"], seg["lat"]
    mask = (
        (lo >= gate["lon_min"]) & (lo <= gate["lon_max"]) &
        (la >= gate["lat_min"]) & (la <= gate["lat_max"])
    )
    return la[mask]


def find_nb_sb(circumpolar_results, dp_gate):
    """
    NB = contour complet dont la traversée de Drake est la plus au nord.
    SB = contour complet dont la traversée de Drake est la plus au sud.
    Accepte la porte Drake déjà adaptée à l'étendue des données.
    """
    nb_res, nb_lat_found = None, -np.inf
    sb_res, sb_lat_found = None,  np.inf
    n_crossing = 0

    for r in circumpolar_results:
        lats = lat_in_gate(r["seg"], dp_gate)
        if len(lats) == 0:
            continue
        n_crossing += 1
        mx, mn = float(np.nanmax(lats)), float(np.nanmin(lats))
        if mx > nb_lat_found:
            nb_lat_found, nb_res = mx, r
        if mn < sb_lat_found:
            sb_lat_found, sb_res = mn, r

    if n_crossing == 0:
        raise RuntimeError(
            f"Aucun contour ne traverse la porte Drake "
            f"(lon=[{dp_gate['lon_min']},{dp_gate['lon_max']}], "
            f"lat=[{dp_gate['lat_min']:.1f},{dp_gate['lat_max']:.1f}]).\n"
            "  Cause probable :\n"
            "  • Le champ OBS s'arrête à ~50°S : NB est hors du champ.\n"
            "    → Dans ce cas NB/SAF ne peuvent pas être trouvés sur l'OBS.\n"
            "    → Utiliser le MOD pour NB, ou accepter NB=bord nord du champ.\n"
            "  • Vérifier MIN_LON_COVERAGE et la porte DP_GATE."
        )

    if nb_res is sb_res:
        print("  [WARN] NB et SB sont le même contour — bande ACC très étroite ?")

    in_field_nb = "✓" if nb_lat_found > dp_gate["lat_min"] + 1 else "⚠ bord"
    in_field_sb = "✓" if sb_lat_found < dp_gate["lat_max"] - 1 else "⚠ bord"
    print(f"  NB : level={nb_res['level']:.3f} m  lat≈{nb_lat_found:.1f}°  [{in_field_nb}]")
    print(f"  SB : level={sb_res['level']:.3f} m  lat≈{sb_lat_found:.1f}°  [{in_field_sb}]")
    return nb_res, sb_res


# ================================================================
# BLOC 6 — SCORING DES FRONTS INTERNES (SAF, PF, SACCF)
# ================================================================

def segment_passes_box(seg, box):
    """Vrai si le segment traverse la boîte."""
    lats = lat_in_gate(seg, box)
    return len(lats) > 0


def mean_speed_in_box(seg, speed2d, lon2d, lat2d, box, band_deg=1.0):
    """
    Vitesse moyenne de surface dans la boîte le long du segment.
    Renvoie NaN si speed2d est None ou si aucun point.
    """
    if speed2d is None:
        return np.nan

    lo, la = seg["lon"], seg["lat"]
    mask_seg = (
        (lo >= box["lon_min"]) & (lo <= box["lon_max"]) &
        (la >= box["lat_min"]) & (la <= box["lat_max"])
    )
    if not np.any(mask_seg):
        return np.nan

    lo_b, la_b = lo[mask_seg], la[mask_seg]
    speeds = []
    for xi, yi in zip(lo_b, la_b):
        m = (np.abs(lon2d - xi) <= band_deg) & (np.abs(lat2d - yi) <= band_deg)
        if np.any(m):
            v = np.nanmean(speed2d[m])
            if np.isfinite(v):
                speeds.append(v)
    return np.nanmean(speeds) if speeds else np.nan


def score_contour(r, d, nb_level, sb_level,
                  target_rank,
                  choke_boxes=CHOKE_BOXES,
                  choke_weights=CHOKE_WEIGHTS):
    """
    Score Park-like d'un contour candidat.

    target_rank :
      "north"      → favorise les contours proches du NB  (SAF)
      "mid_north"  → milieu haut de la bande (PF)
      "mid_south"  → milieu bas (SACCF)

    Contrainte choke points (selon d["name"]) :
      OBS  → le contour DOIT traverser UFZ, DP et TAS (3 boîtes requises).
             Raisonnement : SWIR (~46°S) et KP (~50°S) sont souvent au-delà
             du bord nord du champ OBS, donc ces boîtes ne sont pas fiables.
      NEMO → au moins 1 boîte touchée (pas de restriction).

    Score = Σ (traversée × vitesse_normée × poids) / N_boîtes
            + bonus de position méridienne
    """
    seg   = r["seg"]
    level = r["level"]
    sp2d  = d["speed"]
    lo2   = d["lon2d"]
    la2   = d["lat2d"]
    prod  = d.get("name", "MOD")

    # --- Contrainte minimum choke points ---
    constraint = CHOKE_MIN_HIT.get(prod, CHOKE_MIN_HIT["MOD"])
    required   = constraint["required"]
    min_count  = constraint["min_count"]

    boxes_hit = {name for name, box in choke_boxes.items()
                 if segment_passes_box(seg, box)}

    # Vérifier les boîtes obligatoires
    missing_required = [b for b in required if b not in boxes_hit]
    if missing_required:
        return -np.inf   # contour invalide pour ce produit

    # Vérifier le nombre minimum
    if len(boxes_hit) < min_count:
        return -np.inf

    # --- Position relative dans la bande NB–SB ∈ [0, 1] ---
    band = abs(nb_level - sb_level) if abs(nb_level - sb_level) > 0 else 1.0
    rel  = (level - sb_level) / band

    target = {"north": 0.75, "mid_north": 0.50, "mid_south": 0.25}[target_rank]
    position_bonus = 1.0 - abs(rel - target) * 2

    # --- Score choke points ---
    choke_score = 0.0
    for name, box in choke_boxes.items():
        if name not in boxes_hit:
            continue
        sp_val  = mean_speed_in_box(seg, sp2d, lo2, la2, box)
        sp_norm = min(sp_val / JET_THRESHOLD, 2.0) if np.isfinite(sp_val) else 0.5
        choke_score += choke_weights[name] * sp_norm

    total = choke_score / len(choke_boxes) + max(position_bonus, 0)
    return total


def find_best_front(circumpolar_results, d, nb_level, sb_level,
                    target_rank, margin=0.02):
    """
    Cherche le contour avec le meilleur score Park dans la bande NB–SB.
    Les contours ne respectant pas la contrainte choke points (CHOKE_MIN_HIT)
    sont automatiquement éliminés par score_contour (retourne -inf).
    `margin` (m) exclut les niveaux trop proches de NB ou SB.
    """
    lo_min = min(nb_level, sb_level) + margin
    lo_max = max(nb_level, sb_level) - margin

    candidates = [r for r in circumpolar_results
                  if lo_min <= r["level"] <= lo_max]

    if not candidates:
        print(f"  [WARNING] aucun candidat dans la bande pour {target_rank}")
        return None, np.nan

    best, best_score = None, -np.inf
    n_valid = 0
    for r in candidates:
        sc = score_contour(r, d, nb_level, sb_level, target_rank)
        if sc > -np.inf:
            n_valid += 1
        if sc > best_score:
            best_score, best = sc, r

    prod = d.get("name", "?")
    constraint = CHOKE_MIN_HIT.get(prod, CHOKE_MIN_HIT["MOD"])
    req_str = "+".join(constraint["required"]) if constraint["required"] else "aucune"
    print(f"    [{prod}] {target_rank:10s} : {n_valid}/{len(candidates)} candidats "
          f"valides (choke requis : {req_str})")

    if best_score == -np.inf:
        print(f"  [WARNING] aucun contour ne satisfait la contrainte choke "
              f"pour {target_rank} ({prod}) — relâcher CHOKE_MIN_HIT ?")
        return None, np.nan

    return best, best_score


# ================================================================
# BLOC 7 — INTERPOLATION FRONT → GRILLE LON
# ================================================================

def interp_front(seg, lon_grid=LON_GRID, band=SEARCH_BAND_DEG):
    if seg is None:
        return np.full(len(lon_grid), np.nan)
    lo, la = seg["lon"].copy(), seg["lat"].copy()
    idx = np.argsort(lo); lo, la = lo[idx], la[idx]
    out = np.full(len(lon_grid), np.nan)
    for i, x in enumerate(lon_grid):
        sel = np.abs(lo - x) <= band
        if np.any(sel):
            out[i] = np.nanmedian(la[sel])
    return out


# ================================================================
# BLOC 8 — LARGEURS + CHOKE POINTS
# ================================================================

def lat_to_km(a, b):
    return 111.2 * np.abs(np.asarray(a, float) - np.asarray(b, float))


def compute_widths(fronts):
    return {
        "ACC_km":    lat_to_km(fronts["NB"],   fronts["SB"]),
        "SAF_SACCF": lat_to_km(fronts["SAF"],  fronts["SACCF"]),
        "SAF_PF":    lat_to_km(fronts["SAF"],  fronts["PF"]),
        "PF_SACCF":  lat_to_km(fronts["PF"],   fronts["SACCF"]),
    }


def detect_choke_points(width_km, lon_grid=LON_GRID,
                         prominence=80, min_dist=10):
    arr = np.array(width_km, float)
    fill = np.nanmean(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0
    sm   = gaussian_filter1d(np.where(np.isfinite(arr), arr, fill),
                             sigma=2, mode="nearest")
    peaks, _ = find_peaks(-sm, prominence=prominence, distance=min_dist)
    return lon_grid[peaks], sm[peaks], sm


def max_speed_acc(speed2d, lon2d, lat2d, nb_lat, sb_lat, lon_grid=LON_GRID):
    if speed2d is None:
        return None
    out = np.full(len(lon_grid), np.nan)
    for i, x in enumerate(lon_grid):
        m_lo  = np.abs(lon2d - x) <= 1.0
        ln, ls = nb_lat[i], sb_lat[i]
        if not (np.isfinite(ln) and np.isfinite(ls)):
            continue
        m = m_lo & (lat2d >= min(ln, ls)) & (lat2d <= max(ln, ls))
        if np.any(m):
            out[i] = np.nanmax(speed2d[m])
    return out


# ================================================================
# BLOC 9 — FIGURES
# ================================================================

os.makedirs(OUTDIR, exist_ok=True)


def _ax_map(lat_north=-35):
    """
    Projection SouthPolarStereo identique à l'exemple de référence.
    Le masque circulaire utilise sin/cos + center/radius comme dans l'exemple.
    """
    proj = ccrs.SouthPolarStereo()
    ax   = plt.axes(projection=proj)
    ax.set_extent([-180, 180, -90, lat_north], crs=ccrs.PlateCarree())

    # Masque circulaire — exactement comme l'exemple de référence
    theta         = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.coastlines(resolution="110m", color="black", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.gridlines(linewidth=0.4, linestyle="--", color="grey")

    return ax, ccrs.PlateCarree()


def _pline(ax, tr, lo, la, **kw):
    ax.plot(lo, la, transform=tr, **kw)


def _plot_front_clipped(ax, tr, lon_grid, lat_arr, lat_max_data, **kw):
    """
    Trace un front en masquant les portions hors du champ de données
    (lat > lat_max_data). La ligne s'interrompt et reprend naturellement.
    """
    lo = lon_grid.copy().astype(float)
    la = lat_arr.copy().astype(float)
    la[la > lat_max_data] = np.nan
    ax.plot(lo, np.ma.masked_invalid(la), transform=tr, **kw)


def fig_mean_map(d, fronts, label):
    """
    Carte du champ moyen + fronts SAF/PF/SACCF.
    Projection identique à l'exemple de référence.
    - OBS : limitée à -50°S, fronts coupés au bord nord du champ
    - MOD : limitée à -35°S
    """
    import matplotlib.path as mpath

    is_obs       = (label.upper() == "OBS")
    lat_north    = -50 if is_obs else -35
    lat_max_data = d.get("lat_max_eff", LAT_MAX)

    fig = plt.figure(figsize=(10, 10))
    ax, tr = _ax_map(lat_north=lat_north)

    # --- fond : champ moyen ---
    f    = d["field"]
    v0, v1 = np.nanpercentile(f, [2, 98])
    pcm  = ax.pcolormesh(
        d["lon2d"], d["lat2d"], f,
        transform=tr, cmap="RdYlBu_r",
        shading="nearest", vmin=v0, vmax=v1
    )
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.55, pad=0.05)
    cbar.set_label("m", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # --- isohypses fins (tous les 0.1 m) ---
    step = 0.1
    lvls = np.arange(np.floor(v0 / step) * step,
                     np.ceil(v1  / step) * step + step, step)
    ax.contour(
        d["lon2d"], d["lat2d"], f,
        levels=lvls, colors="k", linewidths=0.5, transform=tr
    )

    # --- fronts : SAF, PF, SACCF uniquement ---
    for fn in ["SAF", "PF", "SACCF"]:
        la_arr = fronts[fn]
        if not np.any(np.isfinite(la_arr)):
            continue
        st = FRONT_STYLE[fn]
        if is_obs:
            _plot_front_clipped(ax, tr, LON_GRID, la_arr, lat_max_data,
                                color=st["color"], lw=st["lw"], ls=st["ls"],
                                label=fn, zorder=6)
        else:
            ok = np.isfinite(la_arr)
            ax.plot(LON_GRID[ok], la_arr[ok], transform=tr,
                    color=st["color"], lw=st["lw"], ls=st["ls"],
                    label=fn, zorder=6)

    if not is_obs:
        ax.legend(loc="lower left", fontsize=18, framealpha=0.8)
    ax.set_title(
        f"Fronts ACC (SAF / PF / SACCF) – {label}  (Park 2019)\n",
        fontsize=13, fontweight="bold", pad=20
    )
    out = f"{OUTDIR}/fig1_mean_map_{label}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out}")


def fig_widths(widths, cl_acc, cv_acc, cl_m3, cv_m3, label):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.subplots_adjust(hspace=0.06)

    ax = axes[0]
    ax.plot(LON_GRID, widths["ACC_km"], "k", lw=1.5, label="NB–SB (largeur ACC)")
    ax.scatter(cl_acc, cv_acc, s=80, c="red", zorder=5, label="choke points")
    for cl in cl_acc:
        ax.axvline(cl, color="red", lw=0.6, ls="--", alpha=0.5)
    for nm, lo in KNOWN_CHOKE_LON.items():
        ax.axvline(lo, color="steelblue", lw=0.8, ls=":", alpha=0.7)
        ax.text(lo, 0.98, nm, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="steelblue")
    ax.set_ylabel("Largeur ACC (km)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"Largeurs circumpolaires – {label}  (Park 2019)", fontsize=12)

    ax2 = axes[1]
    ax2.plot(LON_GRID, widths["SAF_SACCF"], "k",   lw=1.5, label="SAF–SACCF")
    ax2.plot(LON_GRID, widths["SAF_PF"],    "b--",  lw=1.0, label="SAF–PF")
    ax2.plot(LON_GRID, widths["PF_SACCF"],  "r--",  lw=1.0, label="PF–SACCF")
    ax2.scatter(cl_m3, cv_m3, s=80, c="darkgreen", zorder=5)
    for nm, lo in KNOWN_CHOKE_LON.items():
        ax2.axvline(lo, color="steelblue", lw=0.8, ls=":", alpha=0.7)
    ax2.set_xlabel("Longitude (°)"); ax2.set_ylabel("Distance (km)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    out = f"{OUTDIR}/fig2_widths_{label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[FIG] {out}")



# ================================================================
# BLOC 10 — RAPPORT
# ================================================================


def report(label, fronts, widths, cl_acc, cv_acc, cl_m3, cv_m3,
           nb_lv, saf_lv, pf_lv, saccf_lv, sb_lv):
    print(f"\n{'='*60}")
    print(f"  RAPPORT — {label}")
    print(f"{'='*60}")

    # --- Niveaux ---
    def _lv(v): return f"{v:.3f}" if np.isfinite(float(v)) else "NaN"
    print(f"  Niveaux retenus (m) :")
    print(f"    NB={_lv(nb_lv)}  SAF={_lv(saf_lv)}  PF={_lv(pf_lv)}"
          f"  SACCF={_lv(saccf_lv)}  SB={_lv(sb_lv)}")

    # --- Valeurs aux sites clés ---
    for site, lo_site in [("UFZ (~144°W)", -144), ("Drake (~62°W)", -62)]:
        i = np.argmin(np.abs(LON_GRID - lo_site))
        print(f"\n  {site}")
        for fn in ["NB", "SAF", "PF", "SACCF", "SB"]:
            v = fronts[fn][i]
            txt = f"{v:.1f}°S" if np.isfinite(v) else "NaN"
            print(f"    {fn:6s} = {txt}")
        acc_w = widths["ACC_km"][i]
        m3_w  = widths["SAF_SACCF"][i]
        print(f"    ACC largeur (NB–SB)  = "
              f"{acc_w:.0f} km" if np.isfinite(acc_w) else "    ACC largeur = NaN")
        print(f"    Conc. fronts (SAF–SACCF) = "
              f"{m3_w:.0f} km" if np.isfinite(m3_w) else "    SAF–SACCF = NaN")

    # --- Choke points ACC (NB–SB) ---
    print(f"\n  Choke points ACC (min largeur NB–SB) :")
    if len(cl_acc) == 0:
        print("    aucun détecté (largeur ACC trop souvent NaN ?)")
    else:
        for lon_c, w_c in zip(cl_acc, cv_acc):
            print(f"    lon={lon_c:+7.1f}°   largeur ACC ≈ {w_c:.0f} km")

    # --- Choke points concentration grands fronts (SAF–SACCF) ---
    print(f"\n  Choke points concentration (min SAF–SACCF) :")
    if len(cl_m3) == 0:
        print("    aucun détecté")
    else:
        for lon_c, w_c in zip(cl_m3, cv_m3):
            print(f"    lon={lon_c:+7.1f}°   SAF–SACCF ≈ {w_c:.0f} km")

    print(f"{'='*60}\n")


# ================================================================
# BLOC 12 — PIPELINE PRINCIPAL
# ================================================================

def process(raw, label):
    print(f"\n{'─'*60}")
    print(f"  Pipeline Park – {label}")
    print(f"{'─'*60}")

    # 0. Masque latitude (détecte l'étendue réelle des données)
    d = mask_latitudes(raw)
    lat_s = d["lat_min_eff"]
    lat_n = d["lat_max_eff"]
    print(f"  Champ effectif : lat=[{lat_s:.1f}, {lat_n:.1f}]")

    # Avertissement si le champ ne couvre pas toute la bande ACC
    if lat_n > -48:
        print(f"  [WARN] Le champ s'arrête à {lat_n:.1f}°S — NB/SAF peuvent être hors champ.")
    if lat_s > -58:
        print(f"  [WARN] Le champ s'arrête à {lat_s:.1f}°S — SB/SACCF peuvent être hors champ.")

    # 1. Scan de tous les niveaux
    circ_results = scan_all_levels(d)
    if not circ_results:
        raise RuntimeError(f"[{label}] aucun contour circumpolaire trouvé !")

    # 2. Adapter la porte Drake à l'étendue des données
    dp_gate = adapt_dp_gate(d)

    # 3. NB / SB à Drake Passage
    print("  Recherche NB/SB à Drake Passage…")
    nb_res, sb_res = find_nb_sb(circ_results, dp_gate)
    nb_lv, sb_lv   = nb_res["level"], sb_res["level"]

    # 4. SAF, PF, SACCF
    print("  Sélection SAF (target=north dans bande)…")
    saf_res, saf_sc  = find_best_front(circ_results, d, nb_lv, sb_lv, "north")

    print("  Sélection PF  (target=mid_north)…")
    pf_res,  pf_sc   = find_best_front(circ_results, d, nb_lv, sb_lv, "mid_north")

    print("  Sélection SACCF (target=mid_south)…")
    sc_res,  sc_sc   = find_best_front(circ_results, d, nb_lv, sb_lv, "mid_south")

    saf_lv   = saf_res["level"] if saf_res else np.nan
    pf_lv    = pf_res["level"]  if pf_res  else np.nan
    saccf_lv = sc_res["level"]  if sc_res  else np.nan
    print(f"  → SAF={saf_lv:.3f} m  (score={saf_sc:.3f})")
    print(f"  → PF ={pf_lv:.3f} m  (score={pf_sc:.3f})")
    print(f"  → SACCF={saccf_lv:.3f} m  (score={sc_sc:.3f})")

    # 5. Interpoler sur grille lon
    fronts = {
        "NB":    interp_front(nb_res["seg"]),
        "SAF":   interp_front(saf_res["seg"] if saf_res else None),
        "PF":    interp_front(pf_res["seg"]  if pf_res  else None),
        "SACCF": interp_front(sc_res["seg"]  if sc_res  else None),
        "SB":    interp_front(sb_res["seg"]),
    }

    # 6. Largeurs
    widths = compute_widths(fronts)

    # 7. Choke points
    cl_acc, cv_acc, _ = detect_choke_points(widths["ACC_km"])
    cl_m3, cv_m3,  _ = detect_choke_points(widths["SAF_SACCF"], prominence=50)

    # filtre vitesse
    sp_acc = max_speed_acc(d["speed"], d["lon2d"], d["lat2d"],
                            fronts["NB"], fronts["SB"])
    if sp_acc is not None:
        keep = np.array([
            sp_acc[np.argmin(np.abs(LON_GRID - cl))] >= JET_THRESHOLD
            for cl in cl_acc
        ])
        cl_acc, cv_acc = cl_acc[keep], cv_acc[keep]

    # 8. Figures
    fig_mean_map(d, fronts, label)
    fig_widths(widths, cl_acc, cv_acc, cl_m3, cv_m3, label)

    # 9. Rapport
    report(label, fronts, widths, cl_acc, cv_acc, cl_m3, cv_m3,
           nb_lv, saf_lv, pf_lv, saccf_lv, sb_lv)

    return fronts, widths, cl_acc


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*60)
    print("  park_acc_fronts_v3.py  –  Park et al. 2019")
    print("="*60)
    os.makedirs(OUTDIR, exist_ok=True)

    # OBS
    try:
        obs = load_obs(OBS_FILE)
        process(obs, "OBS")
    except FileNotFoundError as e:
        print(f"[SKIP OBS] {e}")

    # MOD
    try:
        mod = load_mod(MOD_FILE)
        process(mod, "MOD")
    except FileNotFoundError as e:
        print(f"[SKIP MOD] {e}")

    print(f"\n[DONE] Sorties dans : {OUTDIR}/")


if __name__ == "__main__":
    main()