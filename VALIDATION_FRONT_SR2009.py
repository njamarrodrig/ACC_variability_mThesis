"""
sr2009_vs_park_NEMO.py
======================
Comparaison de DEUX methodes de definition des fronts de l'ACC, appliquees
au MEME jeu de donnees (NEMO, ZOS mensuel 1/4 deg, 1991-2023) :

  (1) Sokolov & Rintoul (2009a)  -- identifiee ici, sur les 396 cartes
      MENSUELLES, pour exploiter la repetition statistique des maxima de
      |grad eta| comme dans la methode originale.
  (2) Park et al. (2019)         -- importee depuis FRONT_PARK_OBSNEMO.py
      (pipeline MOD uniquement).

Le but n'est PAS de valider contre la litterature, mais d'isoler l'effet du
CHOIX METHODOLOGIQUE, a donnees constantes.

Methode S&R 2009a appliquee ici
-------------------------------
  Pour CHAQUE mois t :
    1. |grad eta| en m/100 km ; points actifs > 0.25 m/100 km OU max local.
    2. Par secteur (36 secteurs de 10 deg) : histogramme SSH pondere par
       |grad eta| -> pics ; assignation SAF/PF/SACCF (cf. ci-dessous).
  Agregation :
    3. label de secteur = mediane des zeta_i sur les 396 mois.
    4. exclusion des secteurs aberrants (> SIGMA_REJECT sigma).
    5. label circumpolaire = mediane des secteurs retenus.
    6. chemin du front = contour SSH = label sur le champ ZOS MOYEN.

Assignation SAF / PF / SACCF -- FENETRE D'ECART SSH
---------------------------------------------------
  Cette etude ne detecte qu'UN SAF et UN PF (pas les 3 branches de chacun
  comme S&R). L'ecart pertinent est donc celui entre les CENTRES de famille
  SAF et PF, plus large que l'ecart entre branches medianes.

  La cascade brute "PF = pic le plus proeminent sous le SAF" attrape une
  BRANCHE SUD du SAF -> tout est decale d'un cran vers le sud. On impose donc
  une FENETRE d'ecart SSH entre fronts adjacents :
    PF    cherche dans [SAF - GAP_MAX, SAF - GAP_MIN]
    SACCF cherche dans [PF  - GAP_MAX, PF  - GAP_MIN]
  GAP_MIN empeche le PF d'etre colle au SAF (saute les branches du SAF) ;
  GAP_MAX empeche le PF de sauter jusqu'au SACCF. Centre de fenetre ~0.35 m
  (ecart SAF-PF typique entre familles).

Fenetres de recherche par secteur (SECTOR_OVERRIDES)
----------------------------------------------------
  Pour quelques secteurs en aval des injections subtropicales, on peut
  restreindre la recherche d'un front a une fenetre de LATITUDE physique.
  Ces overrides sont prioritaires sur la fenetre d'ecart SSH.

Trace des cartes
----------------
  Fronts S&R ET Park traces par ax.contour du champ ZOS :
    S&R  -> contour au label circumpolaire detecte ;
    Park -> contour aux niveaux importes du pipeline Park.
  La comparaison quantitative (biais, r) utilise les trajectoires reelles
  du pipeline Park.

Sorties (dossier ./outputs_sr2009_v2/) :
  - sr2009_NEMO_polar_grad.png    - sr2009_vs_park_NEMO_polar.png
  - sr2009_vs_park_latbylon.png   - sr2009_labels.csv
  - sr2009_sector_latitudes.csv   - sr2009_vs_park_metrics.csv

Usage : python sr2009_vs_park_NEMO.py
"""

import os
import csv
import warnings
import importlib.util
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

# =============================================================================
# PARAMETRES  --  A ADAPTER A TON ARBORESCENCE
# =============================================================================

FILEPATH         = "zos_1991_2023_IS.nc"
MASK_FILE        = "./Transport_sv/mask_continent.nc"
PARK_MODULE_PATH = "/cyfast/njamar/SSH/FRONT_PARK_OBSNEMO.py"
OUTPUT_DIR       = "./outputs_sr2009_v2/"

LAT_MIN = -70.0          # deg S
LAT_MAX = -35.0          # deg S

N_SECTORS      = 36
LON_STEP       = 360.0 / N_SECTORS
GRAD_THRESHOLD = 0.25                    # m/100 km  (S&R 2009a)
R_EARTH        = 6_371_000.0

# --- Fenetre d'ecart SSH entre fronts adjacents ------------------------------
# Le PF est cherche dans [SAF - GAP_MAX, SAF - GAP_MIN] ; idem SACCF / PF.
# Centre de fenetre ~ 0.35 m (ecart SAF-PF typique entre familles de fronts).
GAP_MIN = 0.20           # m  -- ecart minimal (saute les branches du SAF)
GAP_MAX = 0.55           # m  -- ecart maximal (n'attrape pas le front suivant)

# --- Exclusion des secteurs aberrants ----------------------------------------
SIGMA_REJECT     = 2.0
MAX_REJECT_ITER  = 3
MIN_SECTORS_KEEP = 8

# Niveaux SSH attendus pour Park (controle vs pipeline).
PARK_LEVELS_EXPECTED = {"SAF": -0.620, "PF": -0.960, "SACCF": -1.280}
PARK_LEVEL_TOL       = 0.05      # m

# -----------------------------------------------------------------------------
# FENETRES DE RECHERCHE PAR SECTEUR (latitude). Prioritaires sur la fenetre
# d'ecart SSH. Mettre {} pour S&R + fenetre d'ecart partout.
# -----------------------------------------------------------------------------
SECTOR_OVERRIDES = {
    (0.0, 30.0): {
        "SAF":   (-50.0, -43.0),
        "PF":    (-52.0, -46.0),
        "SACCF": (-56.0, -50.0),
    },
    (30.0, 60.0): {
        "SAF": (-50.0, -42.0),
        "PF":  (-56.0, -48.0),
    },
    (190.0, 210.0): {
        "SAF": (-58.0, -51.0),
        "PF":  (-62.0, -54.0),
    },
    (240.0, 270.0): {
        "SAF":   (-60.0, -52.0),
        "PF":    (-65.0, -52.0),
        "SACCF": (-70.0, -63.0),
    },
    (330.0, 360.0): {
        "SAF": (-51.0, -44.0),
        "PF":  (-55.0, -45.0),
    },
}

SR_STYLES = {
    "SAF":   {"color": "black", "lw": 2.4, "ls": "-"},
    "PF":    {"color": "black", "lw": 2.4, "ls": "--"},
    "SACCF": {"color": "black", "lw": 2.4, "ls": ":"},
}
PARK_COLORS = {"SAF": "crimson", "PF": "royalblue", "SACCF": "forestgreen"}
PARK_LW     = 2.0

FRONTS = ["SAF", "PF", "SACCF"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PARTIE 0  --  IMPORT DU MODULE PARK
# =============================================================================

def import_park_module(path):
    """Importe FRONT_PARK_OBSNEMO.py comme module (sans declencher son main)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Module Park introuvable : {path}\n"
            "  -> Corriger PARK_MODULE_PATH en haut de ce script."
        )
    spec = importlib.util.spec_from_file_location("FRONT_PARK", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# PARTIE 1  --  SECTEURS ET FENETRES DE RECHERCHE
# =============================================================================

def sector_masks(nav_lon):
    """Precalcule les masques booleens des 36 secteurs (longitude 0-360)."""
    lon_centers = np.arange(LON_STEP / 2, 360, LON_STEP)
    masks = []
    for lon_c in lon_centers:
        lon_w = lon_c - LON_STEP / 2
        lon_e = lon_c + LON_STEP / 2
        if lon_w < 0:
            m = (nav_lon >= lon_w + 360) | (nav_lon < lon_e)
        elif lon_e > 360:
            m = (nav_lon >= lon_w) | (nav_lon < lon_e - 360)
        else:
            m = (nav_lon >= lon_w) & (nav_lon < lon_e)
        masks.append(m)
    return masks


def sector_override(lon_c):
    """Renvoie le dict {front: (lat_sud, lat_nord)} du secteur, ou None."""
    for (lo_min, lo_max), fronts_win in SECTOR_OVERRIDES.items():
        if lo_min <= lon_c < lo_max:
            return fronts_win
    return None


def ssh_window_from_lat(zos_t, sm, nav_lat, lat_sud, lat_nord):
    """
    Convertit une fenetre de latitude (lat_sud, lat_nord) en fenetre SSH
    (ssh_min, ssh_max) pour le secteur sm et le mois courant.
    Le SSH decroit vers le sud. (np.nan, np.nan) si une bande est vide.
    """
    half = 0.5
    band_n = (sm & (nav_lat >= lat_nord - half) & (nav_lat <= lat_nord + half)
              & np.isfinite(zos_t))
    band_s = (sm & (nav_lat >= lat_sud - half)  & (nav_lat <= lat_sud + half)
              & np.isfinite(zos_t))
    if not band_n.any() or not band_s.any():
        return np.nan, np.nan
    ssh_n = float(np.nanmean(zos_t[band_n]))
    ssh_s = float(np.nanmean(zos_t[band_s]))
    return min(ssh_s, ssh_n), max(ssh_s, ssh_n)


# =============================================================================
# PARTIE 2  --  DETECTION DES PICS ET ASSIGNATION
# =============================================================================

def sector_peak_list(ssh_pts, grad_pts):
    """
    Histogramme SSH pondere par |grad eta| sur les points actifs d'un secteur.
    Renvoie (peak_ssh, peak_prominence) -- TOUS les pics detectes.
    """
    if ssh_pts.size < 20:
        return np.array([]), np.array([])

    lo = float(np.percentile(ssh_pts, 3))
    hi = float(np.percentile(ssh_pts, 97))
    if hi <= lo:
        return np.array([]), np.array([])

    bins     = np.linspace(lo, hi, 301)
    bin_ctrs = 0.5 * (bins[:-1] + bins[1:])
    rng      = (ssh_pts >= lo) & (ssh_pts <= hi)
    hist, _  = np.histogram(ssh_pts[rng], bins=bins, weights=grad_pts[rng])
    hist_sm  = uniform_filter1d(hist.astype(float), size=9)
    if hist_sm.max() == 0:
        return np.array([]), np.array([])

    peaks, props = find_peaks(hist_sm,
                              prominence=hist_sm.max() * 0.03,
                              distance=6)
    if peaks.size == 0:
        return np.array([]), np.array([])
    return bin_ctrs[peaks], props["prominences"]


def assign_fronts(peak_ssh, peak_prom, windows=None):
    """
    Assigne SAF / PF / SACCF aux pics d'un secteur.

    Regle de base -- FENETRE D'ECART SSH (cascade ancree) :
      SAF   = pic le plus proeminent (eventuellement borne par son override).
      PF    = pic le plus proeminent dans [SAF - GAP_MAX, SAF - GAP_MIN].
      SACCF = pic le plus proeminent dans [PF  - GAP_MAX, PF  - GAP_MIN].
    La fenetre [gap_min, gap_max] empeche a la fois le PF d'etre colle au SAF
    (branche sud du SAF) et de sauter jusqu'au SACCF.

    Override de secteur (windows = {front: (ssh_min, ssh_max)}) :
      pour les fronts presents dans le dict, la recherche du pic est restreinte
      a (ssh_min, ssh_max). L'override est PRIORITAIRE sur la fenetre d'ecart :
      si un front a un override, sa fenetre d'ecart n'est pas appliquee ; les
      fronts suivants se calent alors sur le front reellement trouve.

    Renvoie [zeta_SAF, zeta_PF, zeta_SACCF] (SSH decroissant, NaN si absent).
    """
    out = np.array([np.nan, np.nan, np.nan])
    if peak_ssh.size == 0:
        return out
    windows = windows or {}

    def strongest(mask):
        if not np.any(mask):
            return np.nan
        p = np.where(mask, peak_prom, -np.inf)
        return float(peak_ssh[int(np.argmax(p))])

    def has_override(name):
        return name in windows and all(np.isfinite(windows[name]))

    # --- SAF ---
    if has_override("SAF"):
        smin, smax = windows["SAF"]
        m_saf = (peak_ssh >= smin) & (peak_ssh <= smax)
    else:
        m_saf = np.ones(peak_ssh.shape, bool)
    saf = strongest(m_saf)
    if not np.isfinite(saf):
        return out

    # --- PF ---
    if has_override("PF"):
        smin, smax = windows["PF"]
        m_pf = (peak_ssh >= smin) & (peak_ssh <= smax)
    else:
        # fenetre d'ecart relative au SAF
        m_pf = ((peak_ssh <= saf - GAP_MIN) & (peak_ssh >= saf - GAP_MAX))
    pf = strongest(m_pf)

    # --- SACCF ---
    ref = pf if np.isfinite(pf) else saf
    if has_override("SACCF"):
        smin, smax = windows["SACCF"]
        m_sc = (peak_ssh >= smin) & (peak_ssh <= smax)
    else:
        m_sc = ((peak_ssh <= ref - GAP_MIN) & (peak_ssh >= ref - GAP_MAX))
    saccf = strongest(m_sc)

    out[:] = [saf, pf, saccf]
    return out


def robust_circumpolar_label(zeta_sec, lon_centers, front_name):
    """
    Label circumpolaire en ECARTANT les secteurs aberrants (> SIGMA_REJECT
    sigma de la mediane). Iteratif, jamais sous MIN_SECTORS_KEEP secteurs.
    Renvoie (label, keep_mask, rejected_lon).
    """
    keep = np.isfinite(zeta_sec)
    rejected_lon = []

    for _ in range(MAX_REJECT_ITER):
        vals = zeta_sec[keep]
        if vals.size <= MIN_SECTORS_KEEP:
            break
        med = float(np.median(vals))
        sig = float(np.std(vals))
        if sig <= 1e-9:
            break
        far = keep & (np.abs(zeta_sec - med) > SIGMA_REJECT * sig)
        if not far.any():
            break
        n_after = int((keep & ~far).sum())
        if n_after < MIN_SECTORS_KEEP:
            order   = np.argsort(-np.abs(zeta_sec - med))
            allowed = keep.sum() - MIN_SECTORS_KEEP
            far = np.zeros_like(keep)
            cnt = 0
            for idx in order:
                if cnt >= allowed:
                    break
                if keep[idx] and abs(zeta_sec[idx] - med) > SIGMA_REJECT * sig:
                    far[idx] = True
                    cnt += 1
            if not far.any():
                break
        for idx in np.where(far)[0]:
            rejected_lon.append(float(lon_centers[idx]))
        keep = keep & ~far

    vals  = zeta_sec[keep]
    label = float(np.median(vals)) if vals.size else np.nan

    if rejected_lon:
        rl = ", ".join(f"{x:.0f}" for x in sorted(rejected_lon))
        print(f"   {front_name:6s} : {len(rejected_lon)} secteur(s) ecarte(s) "
              f"(lon = {rl})")
    else:
        print(f"   {front_name:6s} : aucun secteur ecarte")

    return label, keep, sorted(rejected_lon)


# =============================================================================
# PARTIE 3  --  IDENTIFICATION S&R 2009a SUR LES CARTES MENSUELLES
# =============================================================================

def identify_sr2009(filepath, mask_file):
    """Identifie les fronts S&R 2009a sur les cartes mensuelles NEMO."""
    print("=" * 64)
    print(f"  IDENTIFICATION S&R 2009a  --  {N_SECTORS} secteurs x {LON_STEP:.0f} deg")
    n_ov = len(SECTOR_OVERRIDES)
    print(f"  Fenetre d'ecart SSH : PF/SACCF dans "
          f"[front_nord - {GAP_MAX}, front_nord - {GAP_MIN}] m")
    print(f"  Overrides de latitude : {n_ov} secteur(s) (prioritaires)")
    print(f"  Exclusion secteurs aberrants : ecart > {SIGMA_REJECT} sigma")
    print("=" * 64)

    # --- chargement SSH + masque continent ---
    print("\n[1/5] Chargement SSH + masque continent...")
    ds      = xr.open_dataset(filepath)
    zos     = ds["zos"]
    nav_lat = ds["nav_lat"].values
    nav_lon = ds["nav_lon"].values
    nav_lon = np.where(nav_lon < 0, nav_lon + 360, nav_lon)

    ds_mask = xr.open_dataset(mask_file)
    umask   = ds_mask["umask"]
    if "time_counter" in umask.dims:
        umask = umask.isel(time_counter=0)
    depth_dim  = [d for d in umask.dims if d not in ("y", "x")][0]
    ocean_mask = (umask.isel({depth_dim: 0}).values != 0)
    if ocean_mask.shape != nav_lat.shape:
        print("   ATTENTION : masque incompatible -- ignore")
        ocean_mask = np.ones(nav_lat.shape, dtype=bool)
    else:
        print(f"   Masque OK -- {ocean_mask.sum():,} points ocean")

    base_band = (nav_lat >= LAT_MIN) & (nav_lat <= LAT_MAX)

    dy = np.gradient(np.deg2rad(nav_lat), axis=0) * R_EARTH
    dx = (np.gradient(np.deg2rad(nav_lon), axis=1)
          * R_EARTH * np.cos(np.deg2rad(nav_lat)))

    secmasks    = sector_masks(nav_lon)
    lon_centers = np.arange(LON_STEP / 2, 360, LON_STEP)
    overrides   = [sector_override(lc) for lc in lon_centers]
    n_months    = zos.sizes["time_counter"]
    print(f"   {n_months} cartes mensuelles a traiter "
          f"(approche S&R : detection mois par mois).")

    # --- boucle mensuelle ---
    print("\n[2/5] Detection des labels SSH mois par mois...")
    labels = {f: np.full((N_SECTORS, n_months), np.nan) for f in FRONTS}
    lat_of = {f: np.full((N_SECTORS, n_months), np.nan) for f in FRONTS}
    grad_sum = np.zeros(nav_lat.shape, dtype=float)
    grad_cnt = np.zeros(nav_lat.shape, dtype=float)

    for t in range(n_months):
        zos_t = zos.isel(time_counter=t).values
        zos_t = np.where(ocean_mask, zos_t, np.nan)
        in_band_t = base_band & np.isfinite(zos_t)

        with np.errstate(invalid="ignore", divide="ignore"):
            deta_dy = np.gradient(zos_t, axis=0) / dy
            deta_dx = np.gradient(zos_t, axis=1) / dx
        grad = np.sqrt(deta_dx**2 + deta_dy**2) * 1e5
        grad = np.where(in_band_t, grad, np.nan)

        grad_sum += np.nan_to_num(grad)
        grad_cnt += np.isfinite(grad)

        local_max = (grad == maximum_filter(
            np.where(np.isfinite(grad), grad, 0.0), size=5))
        active = ((grad > GRAD_THRESHOLD) | local_max) & in_band_t

        for s, sm in enumerate(secmasks):
            pts = active & sm
            if pts.sum() < 20:
                continue
            ssh_pts  = zos_t[pts]
            grad_pts = np.where(np.isfinite(grad[pts]), grad[pts], 0.0)

            peak_ssh, peak_prom = sector_peak_list(ssh_pts, grad_pts)
            if peak_ssh.size == 0:
                continue

            windows = None
            ov = overrides[s]
            if ov is not None:
                windows = {}
                for fname, (lat_s, lat_n) in ov.items():
                    windows[fname] = ssh_window_from_lat(
                        zos_t, sm, nav_lat, lat_s, lat_n)

            top = assign_fronts(peak_ssh, peak_prom, windows)

            for fi, f in enumerate(FRONTS):
                zeta = top[fi]
                labels[f][s, t] = zeta
                if np.isfinite(zeta):
                    sec_lat = nav_lat[pts]
                    idx = np.argsort(np.abs(ssh_pts - zeta))[:50]
                    lat_of[f][s, t] = float(np.median(sec_lat[idx]))

        if (t + 1) % 50 == 0 or (t + 1) == n_months:
            print(f"   ... {t + 1}/{n_months} mois traites")

    # --- agregation : mediane temporelle par secteur ---
    print("\n[3/5] Agregation des labels (mediane temporelle par secteur)...")
    zos_mean  = zos.mean(dim="time_counter").values
    zos_mean  = np.where(ocean_mask, zos_mean, np.nan)
    grad_mean = np.where(grad_cnt > 0, grad_sum / np.maximum(grad_cnt, 1), np.nan)
    in_band   = base_band & np.isfinite(zos_mean)

    zeta_sec_all, sigma_sec_all, sector_lat, n_valid = {}, {}, {}, {}
    for f in FRONTS:
        zeta_sec_all[f] = np.nanmedian(labels[f], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigma_sec_all[f] = np.nanstd(labels[f], axis=1)
        sector_lat[f] = np.nanmedian(lat_of[f], axis=1)
        n_valid[f] = int(np.isfinite(zeta_sec_all[f]).sum())

    # --- exclusion des secteurs aberrants + label circumpolaire ---
    print("\n[4/5] Exclusion des secteurs aberrants et label circumpolaire...")
    zeta_circ, sigma_temporal, sigma_spatial = {}, {}, {}
    n_kept, rejected = {}, {}
    for f in FRONTS:
        label, keep, rej = robust_circumpolar_label(
            zeta_sec_all[f], lon_centers, f)
        zeta_circ[f]      = label
        n_kept[f]         = int(keep.sum())
        rejected[f]       = rej
        sigma_spatial[f]  = (float(np.std(zeta_sec_all[f][keep]))
                             if keep.any() else np.nan)
        sigma_temporal[f] = (float(np.nanmedian(sigma_sec_all[f][keep]))
                             if keep.any() else np.nan)

    print("\n   Front    label SSH      sigma_temp      sigma_spat   "
          "secteurs retenus")
    print("   " + "-" * 64)
    for f in FRONTS:
        print(f"   {f:6s}  {zeta_circ[f]:+9.4f} m   {sigma_temporal[f]:9.4f} m   "
              f"{sigma_spatial[f]:9.4f} m   {n_kept[f]:>3d}/{n_valid[f]}")
    print("\n   sigma_temp = variabilite temporelle dans un secteur "
          "(= barres Fig.3 S&R)")
    print("   sigma_spat = dispersion des labels des secteurs retenus")

    # --- diagnostic : latitude mediane par secteur ---
    print("\n[5/5] Diagnostic -- latitude mediane de chaque front par secteur :")
    print("   (sert a regler SECTOR_OVERRIDES ; '*' = secteur avec override)")
    print(f"   {'lon':>6}  {'SAF':>8}  {'PF':>8}  {'SACCF':>8}")
    for s, lc in enumerate(lon_centers):
        flag = "*" if overrides[s] is not None else " "
        vals = []
        for f in FRONTS:
            v = sector_lat[f][s]
            vals.append(f"{v:8.2f}" if np.isfinite(v) else f"{'--':>8}")
        print(f" {flag} {lc:6.1f}  {vals[0]}  {vals[1]}  {vals[2]}")

    return dict(zeta_circ=zeta_circ, sigma_temporal=sigma_temporal,
                sigma_spatial=sigma_spatial, n_valid=n_valid,
                n_kept=n_kept, rejected=rejected,
                sector_lat=sector_lat, lon_centers=lon_centers,
                zos_mean=zos_mean, grad_mean=grad_mean,
                nav_lon=nav_lon, nav_lat=nav_lat, in_band=in_band)


# =============================================================================
# PARTIE 4  --  CONTOURS -> LATITUDE SUR GRILLE COMMUNE
# =============================================================================

def interp_regular(nav_lon, nav_lat, field, in_band, res=0.25):
    """Interpole un champ curvilineaire sur une grille reguliere 0-360."""
    lon_reg = np.arange(0, 360, res)
    lat_reg = np.arange(LAT_MIN, LAT_MAX + res, res)
    LON_G, LAT_G = np.meshgrid(lon_reg, lat_reg)
    ok  = in_band & np.isfinite(field)
    pts = np.column_stack([nav_lon[ok], nav_lat[ok]])
    reg = griddata(pts, field[ok], (LON_G, LAT_G), method="linear")
    return lon_reg, lat_reg, reg


def _extract_paths(cs):
    """Compatibilite matplotlib < 3.8 et >= 3.8."""
    try:
        return cs.get_paths()
    except AttributeError:
        return cs.collections[0].get_paths() if cs.collections else []


def extract_front_latitude(zos_reg, lon_reg, lat_reg, level, lon_grid, pad=8):
    """
    Contour SSH = level sur le champ ZOS regulier -> tableau latitude(longitude)
    sur lon_grid. Grille periodique en longitude, tous segments concatenes,
    mediane des latitudes par classe de longitude.
    Sert UNIQUEMENT a la comparaison quantitative S&R.
    """
    out = np.full(np.asarray(lon_grid).shape, np.nan)
    if not np.isfinite(level):
        return out

    lon_ext = np.concatenate([lon_reg[-pad:] - 360, lon_reg, lon_reg[:pad] + 360])
    zos_ext = np.concatenate([zos_reg[:, -pad:], zos_reg, zos_reg[:, :pad]], axis=1)
    LONe, LATe = np.meshgrid(lon_ext, lat_reg)

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(LONe, LATe, zos_ext, levels=[level])
    except Exception:
        plt.close(fig)
        return out
    paths = _extract_paths(cs)
    plt.close(fig)

    lo_all, la_all = [], []
    for p in paths:
        v  = p.vertices
        lo, la = v[:, 0], v[:, 1]
        ok = np.isfinite(lo) & np.isfinite(la)
        lo, la = lo[ok], la[ok]
        if lo.size >= 30:
            lo_all.append(lo)
            la_all.append(la)
    if not lo_all:
        return out
    lons = np.concatenate(lo_all)
    lats = np.concatenate(la_all)

    lon_grid = np.asarray(lon_grid, float)
    if np.nanmin(lon_grid) < 0.0:
        lons = ((lons + 180.0) % 360.0) - 180.0
    else:
        lons = lons % 360.0

    order = np.argsort(lon_grid)
    lon_s = lon_grid[order]
    edges = np.empty(lon_s.size + 1)
    edges[1:-1] = 0.5 * (lon_s[:-1] + lon_s[1:])
    edges[0]    = lon_s[0]  - 0.5 * (lon_s[1]  - lon_s[0])
    edges[-1]   = lon_s[-1] + 0.5 * (lon_s[-1] - lon_s[-2])

    idx   = np.digitize(lons, edges) - 1
    valid = (idx >= 0) & (idx < lon_s.size) & np.isfinite(lats)
    iv, lv = idx[valid], lats[valid]
    for k in range(lon_s.size):
        sel = (iv == k)
        if sel.any():
            out[order[k]] = np.median(lv[sel])
    return out


# =============================================================================
# PARTIE 5  --  FRONTS PARK 2019 SUR NEMO
# =============================================================================

def compute_park_fronts(park):
    """
    Execute le pipeline Park (FRONT_PARK_OBSNEMO.py) pour le MODELE NEMO.
    Renvoie : (fronts, levels). Verifie les niveaux vs PARK_LEVELS_EXPECTED.
    """
    print("\n" + "=" * 64)
    print("  FRONTS PARK 2019  --  pipeline applique a NEMO (MOD)")
    print("=" * 64)

    raw = park.load_mod(park.MOD_FILE)
    d   = park.mask_latitudes(raw)

    circ = park.scan_all_levels(d)
    if not circ:
        raise RuntimeError("[Park] aucun contour circumpolaire trouve.")

    gate = park.adapt_dp_gate(d)
    nb_res, sb_res = park.find_nb_sb(circ, gate)
    nb_lv, sb_lv   = nb_res["level"], sb_res["level"]

    saf_res, _ = park.find_best_front(circ, d, nb_lv, sb_lv, "north")
    pf_res,  _ = park.find_best_front(circ, d, nb_lv, sb_lv, "mid_north")
    sc_res,  _ = park.find_best_front(circ, d, nb_lv, sb_lv, "mid_south")

    fronts = {
        "SAF":   park.interp_front(saf_res["seg"] if saf_res else None),
        "PF":    park.interp_front(pf_res["seg"]  if pf_res  else None),
        "SACCF": park.interp_front(sc_res["seg"]  if sc_res  else None),
    }
    levels = {
        "SAF":   saf_res["level"] if saf_res else np.nan,
        "PF":    pf_res["level"]  if pf_res  else np.nan,
        "SACCF": sc_res["level"]  if sc_res  else np.nan,
    }
    print(f"   Niveaux Park (m) : SAF={levels['SAF']:.3f}  "
          f"PF={levels['PF']:.3f}  SACCF={levels['SACCF']:.3f}")
    for f in FRONTS:
        exp = PARK_LEVELS_EXPECTED.get(f)
        got = levels[f]
        if exp is not None and np.isfinite(got) and abs(got - exp) > PARK_LEVEL_TOL:
            print(f"   [AVERTISSEMENT] niveau Park {f} = {got:+.3f} m, "
                  f"attendu {exp:+.3f} m (ecart > {PARK_LEVEL_TOL} m)")
    return fronts, levels


# =============================================================================
# PARTIE 6  --  COMPARAISON QUANTITATIVE
# =============================================================================

def compare_fronts(sr_fronts, park_fronts):
    """
    Compare les fronts S&R et Park sur les longitudes communes.
    Convention de biais : lat_SR - lat_Park  (>0 : front S&R plus au NORD).
    """
    rows = []
    for f in FRONTS:
        sr = np.asarray(sr_fronts[f],   float)
        pk = np.asarray(park_fronts[f], float)
        n  = min(sr.size, pk.size)
        sr, pk = sr[:n], pk[:n]
        ok = np.isfinite(sr) & np.isfinite(pk)
        n_ok = int(ok.sum())

        row = dict(front=f, n=n_ok, lat_sr=np.nan, lat_park=np.nan,
                   bias=np.nan, rmse=np.nan, r=np.nan)
        if n_ok >= 3:
            a, b = sr[ok], pk[ok]
            row["lat_sr"]   = float(np.mean(a))
            row["lat_park"] = float(np.mean(b))
            row["bias"]     = float(np.mean(a - b))
            row["rmse"]     = float(np.sqrt(np.mean((a - b) ** 2)))
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                row["r"] = float(np.corrcoef(a, b)[0, 1])
        rows.append(row)
    return rows


def print_metrics(rows):
    hdr = (f"{'front':<8}{'N':>5}{'lat_SR':>10}{'lat_Park':>11}"
           f"{'biais':>10}{'RMSE':>9}{'r':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in rows:
        def f(x, d=2):
            return f"{x:.{d}f}" if np.isfinite(x) else "  --"
        print(f"{m['front']:<8}{m['n']:>5}{f(m['lat_sr']):>10}"
              f"{f(m['lat_park']):>11}{f(m['bias']):>10}"
              f"{f(m['rmse']):>9}{f(m['r'], 3):>8}")
    print("-" * len(hdr))
    print("  biais = lat_SR - lat_Park  (>0 : front S&R plus au nord)")
    print("  unites : degres de latitude  |  r : correlation lat(lon)")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[CSV] {path}")


# =============================================================================
# PARTIE 7  --  FIGURES  (fronts traces par ax.contour du champ ZOS)
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
    ax.gridlines(linewidth=0.4, linestyle="--", color="grey", zorder=3)
    return ax


def contour_front(ax, tr, lon_reg, lat_reg, zos_reg, level, style):
    """Trace un front par contour du champ ZOS au niveau `level`."""
    if not np.isfinite(level):
        return
    LON_G, LAT_G = np.meshgrid(lon_reg, lat_reg)
    ax.contour(LON_G, LAT_G, zos_reg, levels=[level],
               colors=[style["color"]], linewidths=style["lw"],
               linestyles=[style["ls"]], transform=tr, zorder=style["z"])


def fig_grad(lon_reg, lat_reg, grad_reg, zos_reg, zeta_circ):
    """Carte polaire : |grad eta| moyen + fronts S&R (contours du ZOS)."""
    tr  = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 10))
    ax  = make_polar_ax(fig, lat_north=-35)

    LON_G, LAT_G = np.meshgrid(lon_reg, lat_reg)
    vmax = np.nanpercentile(grad_reg, 97)
    pcm  = ax.pcolormesh(LON_G, LAT_G, grad_reg, cmap="YlOrRd",
                         vmin=0, vmax=vmax, transform=tr,
                         shading="auto", rasterized=True, zorder=2)
    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", location="right",
                        fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label(r"$|\nabla\eta|$ moyen (m 100 km$^{-1}$)", fontsize=11)

    for f in FRONTS:
        st = dict(SR_STYLES[f], z=8)
        contour_front(ax, tr, lon_reg, lat_reg, zos_reg, zeta_circ[f], st)
        ax.plot([], [], color=st["color"], lw=st["lw"], ls=st["ls"],
                label=f"{f}  ($\\zeta$ = {zeta_circ[f]:.3f} m)")

    ax.legend(fontsize=15, loc="lower left", framealpha=0.9,
              title="Fronts S&R 2009a", title_fontsize=15)
    ax.set_title("Fronts ACC -- S&R (2009a) sur NEMO\n"
                 "Detection sur cartes mensuelles 1991-2023  ·  "
                 f"{N_SECTORS} secteurs x {LON_STEP:.0f} deg",
                 fontsize=12, fontweight="bold", pad=18)

    out = os.path.join(OUTPUT_DIR, "sr2009_NEMO_polar_grad.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")


def fig_compare_polar(lon_reg, lat_reg, zos_reg, zeta_circ, park_levels):
    """Carte polaire : fronts S&R vs Park, tous deux contours du champ ZOS."""
    tr  = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 10))
    ax  = make_polar_ax(fig, lat_north=-35)

    for f in FRONTS:
        park_st = {"color": PARK_COLORS[f], "lw": PARK_LW, "ls": "-", "z": 6}
        contour_front(ax, tr, lon_reg, lat_reg, zos_reg, park_levels[f], park_st)
        sr_st = dict(SR_STYLES[f], z=8)
        contour_front(ax, tr, lon_reg, lat_reg, zos_reg, zeta_circ[f], sr_st)

    handles = []
    for f in FRONTS:
        handles.append(mlines.Line2D([], [], color=PARK_COLORS[f],
                                     lw=PARK_LW, ls="-",
                                     label=f"{f} Park ($\\zeta$={park_levels[f]:+.3f})"))
        st = SR_STYLES[f]
        handles.append(mlines.Line2D([], [], color=st["color"], lw=st["lw"],
                                     ls=st["ls"],
                                     label=f"{f} S&R ($\\zeta$={zeta_circ[f]:+.3f})"))
    ax.legend(handles=handles, loc="lower left", fontsize=12,
              framealpha=0.9, title="Methode", title_fontsize=14)
    ax.set_title("Fronts ACC -- S&R (2009a) vs Park (2019)\n"
                 "Contours du champ ZOS moyen  ·  meme jeu NEMO 1991-2023",
                 fontsize=12, fontweight="bold", pad=18)

    out = os.path.join(OUTPUT_DIR, "sr2009_vs_park_NEMO_polar.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")


def fig_lat_by_lon(lon_grid, sr_fronts, park_fronts, metrics):
    """3 panneaux : latitude(longitude) des deux methodes, par front."""
    met = {m["front"]: m for m in metrics}
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    for ax, f in zip(axes, FRONTS):
        ax.plot(lon_grid, park_fronts[f], color=PARK_COLORS[f],
                lw=2.0, label="Park 2019")
        ax.plot(lon_grid, sr_fronts[f], color="black",
                lw=2.0, ls="--", label="S&R 2009a")
        m = met.get(f, {})
        txt = (f"biais = {m.get('bias', np.nan):+.2f} deg   "
               f"r = {m.get('r', np.nan):.3f}   "
               f"RMSE = {m.get('rmse', np.nan):.2f} deg")
        ax.set_title(f"{f}   --   {txt}", fontsize=10, loc="left")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.legend(fontsize=12, loc="best")

    axes[-1].set_xlabel("Longitude (deg)")
    axes[0].set_xlim(np.nanmin(lon_grid), np.nanmax(lon_grid))
    fig.suptitle("Latitude des fronts ACC selon la longitude -- "
                 "S&R 2009a vs Park 2019 (NEMO)",
                 fontsize=12, fontweight="bold", y=0.995)

    out = os.path.join(OUTPUT_DIR, "sr2009_vs_park_latbylon.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    sr = identify_sr2009(FILEPATH, MASK_FILE)

    print("\n[POST] Interpolation reguliere du champ ZOS et de |grad eta|...")
    lon_reg, lat_reg, zos_reg = interp_regular(
        sr["nav_lon"], sr["nav_lat"], sr["zos_mean"], sr["in_band"])
    _, _, grad_reg = interp_regular(
        sr["nav_lon"], sr["nav_lat"], sr["grad_mean"], sr["in_band"])

    park     = import_park_module(PARK_MODULE_PATH)
    LON_GRID = np.asarray(park.LON_GRID, dtype=float)
    park_fronts, park_levels = compute_park_fronts(park)

    sr_fronts = {
        f: extract_front_latitude(zos_reg, lon_reg, lat_reg,
                                  sr["zeta_circ"][f], LON_GRID)
        for f in FRONTS
    }

    print("\n" + "=" * 64)
    print("  COMPARAISON QUANTITATIVE  --  S&R 2009a vs Park 2019")
    print("=" * 64)
    metrics = compare_fronts(sr_fronts, park_fronts)
    print_metrics(metrics)

    label_rows = [
        dict(front=f,
             label_ssh=sr["zeta_circ"][f],
             sigma_temporal=sr["sigma_temporal"][f],
             sigma_spatial=sr["sigma_spatial"][f],
             n_sectors_kept=sr["n_kept"][f],
             n_sectors_valid=sr["n_valid"][f],
             rejected_lon=";".join(f"{x:.0f}" for x in sr["rejected"][f]),
             park_level=park_levels[f])
        for f in FRONTS
    ]
    write_csv(os.path.join(OUTPUT_DIR, "sr2009_labels.csv"),
              ["front", "label_ssh", "sigma_temporal", "sigma_spatial",
               "n_sectors_kept", "n_sectors_valid", "rejected_lon",
               "park_level"], label_rows)

    lat_rows = []
    for s, lc in enumerate(sr["lon_centers"]):
        row = {"lon_center": round(float(lc), 1)}
        for f in FRONTS:
            v = sr["sector_lat"][f][s]
            row[f] = round(float(v), 3) if np.isfinite(v) else ""
        lat_rows.append(row)
    write_csv(os.path.join(OUTPUT_DIR, "sr2009_sector_latitudes.csv"),
              ["lon_center", "SAF", "PF", "SACCF"], lat_rows)

    write_csv(os.path.join(OUTPUT_DIR, "sr2009_vs_park_metrics.csv"),
              ["front", "n", "lat_sr", "lat_park", "bias", "rmse", "r"],
              metrics)

    print()
    fig_grad(lon_reg, lat_reg, grad_reg, zos_reg, sr["zeta_circ"])
    fig_compare_polar(lon_reg, lat_reg, zos_reg, sr["zeta_circ"], park_levels)
    fig_lat_by_lon(LON_GRID, sr_fronts, park_fronts, metrics)

    print(f"\n[TERMINE] Sorties dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()