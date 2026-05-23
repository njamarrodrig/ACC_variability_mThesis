"""
=============================================================================
VARIABILITÉ DES FRONTS ACC — V6
=============================================================================
Méthodologie rigoureuse (Chapman 2020, Kim & Orsi 2014, Sokolov & Rintoul 2009)

Principe clé :
  - travailler sur les MOYENNES ANNUELLES de SSH → cycle saisonnier
    implicitement absent, aucune désaisonnalisation explicite nécessaire
  - détecter un mode commun large échelle de SSH (dilatation/stérisme)
  - si significatif : produire les résultats AVANT et APRÈS correction
  - ne JAMAIS détendre localement chaque pixel SSH
  - pour NEMO : parler de "tendance large échelle" et non "dérive prouvée"

Outputs dans ./outputs_position/ :
  fronts_annual_spaghetti_OBS.png   (+ _raw si correction appliquée)
  fronts_annual_spaghetti_NEMO.png
  fronts_mean_variability_OBS.png
  fronts_mean_variability_NEMO.png
  fronts_circumpolar_timeseries_OBS.png
  fronts_circumpolar_timeseries_NEMO.png
  summary_OBS.csv
  summary_NEMO.csv

=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import Delaunay
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CFG = {
    # Fichiers
    "obs_file"       : "dot_all_30bmedian_eigen6s4v2_sig3.nc",
    "nemo_ssh_file"  : "zos_1991_2023_IS.nc",
    "nemo_mask_file" : "Transport_sv/mask_continent.nc",

    # Variables OBS
    "obs_ssh_var"    : "dot",
    "obs_mask_var"   : "land_mask",
    "obs_lon_var"    : "longitude",
    "obs_lat_var"    : "latitude",
    "obs_time_var"   : "time",
    "obs_time_origin": "2002-07-01",

    # Variables NEMO
    "nemo_ssh_var"   : "zos",
    "nemo_lon_var"   : "nav_lon",
    "nemo_lat_var"   : "nav_lat",
    "nemo_time_var"  : "time_counter",
    "nemo_mask_var"  : "tmask",

    # Niveaux SSH calibrés (Kim & Orsi / Sokolov & Rintoul)
    "levels_obs"  : {"SAF": -0.980, "PF": -1.340, "SACCF": -1.720},
    "levels_nemo" : {"SAF": -0.620, "PF": -0.960, "SACCF": -1.280},

    # Domaine spatial ACC
    "lat_min"     : -70.0,
    "lat_max"     : -35.0,

    # Grille régulière cible pour NEMO (interpolation Delaunay)
    "regrid_dlon" : 0.5,
    "regrid_dlat" : 0.5,

    # Seuil significativité mode commun
    "pvalue_threshold" : 0.05,

    # Lissage longitudinal des lignes de front (nb de points)
    "smooth_window" : 10,

    # Sorties
    "outdir"  : "outputs_position/",
    "fig_dpi" : 150,
}

FRONTS       = ["SAF", "PF", "SACCF"]
FRONT_COLORS = {"SAF": "#2166ac", "PF": "#d6604d", "SACCF": "#4dac26"}
DEG2KM       = 111.0   # approximation à ~55°S


# =============================================================================
# 1.  CHARGEMENT OBS
# =============================================================================

def load_obs(cfg):
    """
    Charge SSH OBS (DOT), transpose en (time, lat, lon), applique masque terre.
    Grille : longitude(1°) × latitude(0.5°), ordre quelconque dans le fichier.
    Retourne DataArray (time, lat, lon).
    """
    print("\n[OBS] Chargement...")
    ds = xr.open_dataset(cfg["obs_file"], decode_times=False)
    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Dimensions brutes : {dict(ds.dims)}")

    # Convertir le temps (days since origin)
    t_raw = ds[cfg["obs_time_var"]].values
    t0    = np.datetime64(cfg["obs_time_origin"])
    times = t0 + (t_raw * np.timedelta64(1, "D")).astype("timedelta64[D]")

    ssh = (ds[cfg["obs_ssh_var"]]
           .assign_coords({cfg["obs_time_var"]: times})
           .rename({cfg["obs_lon_var"]: "lon",
                    cfg["obs_lat_var"]: "lat",
                    cfg["obs_time_var"]: "time"})
           .transpose("time", "lat", "lon")
           .sel(lat=slice(cfg["lat_min"], cfg["lat_max"])))

    # Normaliser longitudes en [-180, 180]
    lon = ssh["lon"].values
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
        ssh = ssh.assign_coords(lon=lon).sortby("lon")

    # Masque terre via numpy (évite le broadcast xarray)
    if cfg["obs_mask_var"] in ds:
        mk = (ds[cfg["obs_mask_var"]]
              .rename({cfg["obs_lon_var"]: "lon", cfg["obs_lat_var"]: "lat"})
              .transpose("lat", "lon")
              .sel(lat=slice(cfg["lat_min"], cfg["lat_max"])))
        mlon = mk["lon"].values
        if mlon.max() > 180:
            mlon = np.where(mlon > 180, mlon - 360, mlon)
            mk = mk.assign_coords(lon=mlon).sortby("lon")
        mnp  = mk.values   # (lat, lon)
        data = np.where(mnp[np.newaxis, :, :] == 1, np.nan, ssh.values)
        ssh  = xr.DataArray(data, dims=ssh.dims, coords=ssh.coords,
                            attrs={"units": "m", "long_name": "DOT"})

    n_t = len(ssh["time"])
    yrs = f"{str(ssh.time.values[0])[:4]} – {str(ssh.time.values[-1])[:4]}"
    print(f"  shape={ssh.shape} | Période : {yrs} | {n_t} mois")
    print(f"  Grille : lon {ssh.lon.values[0]:.1f}°→{ssh.lon.values[-1]:.1f}°  "
          f"lat {ssh.lat.values[0]:.2f}°→{ssh.lat.values[-1]:.2f}°")
    return ssh


# =============================================================================
# 2.  CHARGEMENT NEMO + INTERPOLATION SUR GRILLE RÉGULIÈRE
# =============================================================================

def load_nemo(cfg):
    """
    Charge ZOS NEMO (curvilinéaire y×x), applique masque, interpole sur
    grille régulière par triangulation Delaunay + coordonnées baryentriques
    (triangulation calculée 1 seule fois, einsum pour tous les pas de temps).
    Retourne DataArray (time, lat, lon).
    """
    import time as _t
    print("\n[NEMO] Chargement SSH curvilinéaire...")

    ds      = xr.open_dataset(cfg["nemo_ssh_file"])
    zos     = ds[cfg["nemo_ssh_var"]]              # (time_counter, y, x)
    nav_lon = ds[cfg["nemo_lon_var"]].values       # (y, x)
    nav_lat = ds[cfg["nemo_lat_var"]].values       # (y, x)
    tvals   = ds[cfg["nemo_time_var"]].values
    n_t     = len(tvals)
    print(f"  {n_t} pas de temps | {str(tvals[0])[:7]} → {str(tvals[-1])[:7]}")

    # ── Masque continent (4D → 2D) ────────────────────────────────────────
    ocean = np.ones(nav_lon.shape, bool)
    try:
        dm = xr.open_dataset(cfg["nemo_mask_file"])
        mv = cfg["nemo_mask_var"]
        if mv not in dm:
            mv = list(dm.data_vars)[0]
        mk = dm[mv]
        for d in mk.dims[:-2]:   # réduction 4D→2D : isel(0) sur toutes sauf (y,x)
            mk = mk.isel({d: 0})
        mk_np = mk.values
        if mk_np.shape == nav_lon.shape:
            ocean = (mk_np == 1)
            print(f"  Masque appliqué : {ocean.sum():,} pts océan")
        else:
            print(f"  ⚠ Masque forme incompatible {mk_np.shape} vs {nav_lon.shape}, ignoré")
    except Exception as e:
        print(f"  ⚠ Masque ignoré ({e})")

    # ── Normaliser longitudes ──────────────────────────────────────────────
    nlon = np.where(nav_lon > 180, nav_lon - 360, nav_lon)

    # ── Sélection bande ACC + océan ────────────────────────────────────────
    band = (nav_lat >= cfg["lat_min"] - 2) & (nav_lat <= cfg["lat_max"] + 2)
    use  = band & ocean
    lons, lats = nlon[use], nav_lat[use]
    print(f"  Points source ACC + océan : {use.sum():,}")

    # ── Grille régulière cible ─────────────────────────────────────────────
    lon_r = np.arange(-180, 180,   cfg["regrid_dlon"])
    lat_r = np.arange(cfg["lat_min"],
                      cfg["lat_max"] + cfg["regrid_dlat"],
                      cfg["regrid_dlat"])
    lg, la   = np.meshgrid(lon_r, lat_r)
    target   = np.column_stack([lg.ravel(), la.ravel()])
    n_lr, n_lonr = len(lat_r), len(lon_r)
    print(f"  Grille cible : {n_lr} × {n_lonr}")

    # ── Triangulation Delaunay (1 fois) ────────────────────────────────────
    t0 = _t.time()
    print(f"  Triangulation Delaunay... ", end="", flush=True)
    tri = Delaunay(np.column_stack([lons, lats]))
    print(f"OK ({_t.time()-t0:.1f}s)")

    # ── Poids baryentriques (1 fois) ───────────────────────────────────────
    t0 = _t.time()
    print(f"  Poids baryentriques... ", end="", flush=True)
    simp  = tri.find_simplex(target)
    valid = simp >= 0
    T     = tri.transform[simp[valid], :2, :]
    t_ref = tri.transform[simp[valid],  2, :]
    b2    = np.einsum('nij,nj->ni', T, target[valid] - t_ref)
    bary  = np.column_stack([b2, 1 - b2.sum(1)])   # (N_v, 3)
    verts = tri.simplices[simp[valid]]              # (N_v, 3)
    print(f"OK ({_t.time()-t0:.1f}s) — {valid.sum():,}/{len(target):,} pts")

    # ── Lecture disque unique ──────────────────────────────────────────────
    t0 = _t.time()
    print(f"  Lecture {n_t} pas de temps... ", end="", flush=True)
    zos_np   = zos.values                                       # (n_t, y, x)
    ssh_flat = zos_np.reshape(n_t, -1)[:, use.ravel()].astype(np.float32)
    print(f"OK ({_t.time()-t0:.0f}s) — {ssh_flat.nbytes/1e6:.0f} MB")

    # ── Interpolation einsum (tout d'un coup) ──────────────────────────────
    t0 = _t.time()
    print(f"  Interpolation einsum... ", end="", flush=True)
    interp_v              = np.einsum('tvk,vk->tv', ssh_flat[:, verts], bary)
    flat_res              = np.full((n_t, len(target)), np.nan, dtype=np.float32)
    flat_res[:, valid]    = interp_v
    ssh_reg               = flat_res.reshape(n_t, n_lr, n_lonr)
    print(f"OK ({_t.time()-t0:.1f}s)")

    da = xr.DataArray(ssh_reg, dims=["time", "lat", "lon"],
                      coords={"time": tvals, "lat": lat_r, "lon": lon_r},
                      attrs={"units": "m", "long_name": "ZOS regrillé"})
    print(f"  shape={da.shape}")
    return da


# =============================================================================
# 3.  MODE COMMUN SSH (dilatation / tendance large échelle)
# =============================================================================

def detect_common_mode(ssh_da, label=""):
    """
    Calcule A(t) = moyenne spatiale pondérée par cos(lat) sur la bande ACC.
    Teste la tendance linéaire.

    Retourne (cm_da, cm_info) :
      cm_da   : DataArray (time,) — le mode commun A(t)
      cm_info : dict avec tendance, p-value, signification
    """
    w  = np.cos(np.deg2rad(ssh_da["lat"]))
    cm = ssh_da.weighted(w).mean(["lat", "lon"])   # (time,)

    t_days = (pd.DatetimeIndex(ssh_da["time"].values)
              - pd.DatetimeIndex(ssh_da["time"].values)[0]).days.astype(float)
    valid  = np.isfinite(cm.values)
    sl, it, r, pv, _ = stats.linregress(t_days[valid], cm.values[valid])

    trend_mm_yr  = sl * 365.25 * 1000
    trend_cm_dec = sl * 365.25 * 100 * 10   # cm/décennie
    signif       = pv < CFG["pvalue_threshold"]

    print(f"\n[{label}] Mode commun SSH (tendance large échelle) :")
    print(f"  Tendance  : {trend_mm_yr:+.2f} mm/an = {trend_cm_dec:+.2f} cm/décennie")
    print(f"  R²        : {r**2:.3f}   p-value : {pv:.4f}")
    if signif:
        print(f"  → SIGNIFICATIF (p < {CFG['pvalue_threshold']}) — "
              f"correction appliquée, résultats produits avant ET après")
    else:
        print(f"  → Non significatif (p ≥ {CFG['pvalue_threshold']}) — "
              f"correction appliquée par précaution mais résultats principaux = bruts")

    return cm, {
        "trend_mm_yr"  : trend_mm_yr,
        "trend_cm_dec" : trend_cm_dec,
        "pvalue"       : pv,
        "r2"           : r**2,
        "significant"  : signif,
        "label"        : label,
    }


def apply_common_mode_correction(ssh_da, cm_da):
    """
    Soustrait UNIQUEMENT l'anomalie temporelle du mode commun :
        ssh_corr(t) = ssh(t) - [A(t) - mean(A)]

    Pourquoi l'anomalie et pas la valeur absolue ?
    Les niveaux SSH calibrés (SAF=-0.980m, etc.) sont définis sur le champ
    SSH brut moyen. Si on soustrait A(t) en valeur absolue, on décale toute
    la SSH et les contours calibrés ne correspondent plus à rien.
    En soustrayant seulement A(t) - mean(A), on :
      - conserve le niveau moyen du champ SSH (les contours restent valides)
      - retire uniquement les fluctuations temporelles grandes échelles
      - préserve les gradients spatiaux et la structure régionale
    """
    cm_anom = cm_da - cm_da.mean("time")   # anomalie : moyenne temporelle = 0
    return ssh_da - cm_anom


# =============================================================================
# 4.  EXTRACTION VECTORISÉE DES FRONTS (sur champ annuel ou mensuel)
# =============================================================================

def _extract_one_field(ssh_2d, lat, level):
    """
    Extrait la position (lat) d'un front sur un champ 2D (lat, lon) pour
    un niveau SSH donné.
    Retourne tableau 1D (lon,) de latitudes, NaN si pas de passage.
    Vectorisé numpy — pas de boucle Python.
    """
    n_lat, n_lon = ssh_2d.shape
    diff     = ssh_2d - level                     # (lat, lon)
    sign_d   = np.diff(np.sign(diff), axis=0)     # (lat-1, lon)
    cross    = sign_d != 0
    d_lo     = diff[:-1, :]
    d_hi     = diff[1:,  :]
    both_ok  = np.isfinite(d_lo) & np.isfinite(d_hi)
    valid    = cross & both_ok
    grad     = np.where(valid, np.abs(np.diff(diff, axis=0)), -np.inf)
    best_k   = np.argmax(grad, axis=0)             # (lon,) — indice lat
    has_any  = np.any(valid, axis=0)               # (lon,)

    li   = np.arange(n_lon)
    dlo  = d_lo[best_k, li]
    dhi  = d_hi[best_k, li]
    den  = dhi - dlo
    den  = np.where(np.abs(den) < 1e-12, np.nan, den)
    frac = -dlo / den
    lat_cross = lat[best_k] + frac * (lat[best_k + 1] - lat[best_k])
    return np.where(has_any, lat_cross, np.nan)


def extract_annual_positions(ssh_da, levels, label=""):
    """
    Calcule les moyennes annuelles de SSH puis extrait les fronts.

    → Cycle saisonnier implicitement absent (on travaille sur les annuelles)
    → Retourne dict { front: DataArray(year, lon) }
    """
    print(f"\n[{label}] Moyennes annuelles + extraction des fronts...")
    ssh_ann = ssh_da.groupby("time.year").mean("time")   # (year, lat, lon)
    years   = ssh_ann["year"].values
    lat     = ssh_ann["lat"].values
    lon     = ssh_ann["lon"].values

    positions = {}
    for front, level in levels.items():
        phi = np.full((len(years), len(lon)), np.nan)
        for i_y, yr in enumerate(years):
            field = ssh_ann.sel(year=yr).values   # (lat, lon)
            phi[i_y] = _extract_one_field(field, lat, level)
        pct = np.isfinite(phi).mean() * 100
        positions[front] = xr.DataArray(
            phi, dims=["year", "lon"],
            coords={"year": years, "lon": lon},
            attrs={"level": level, "front": front, "dataset": label})
        print(f"  {front} : couverture = {pct:.1f}%")

    return positions


def extract_mean_position(ssh_da, levels, label=""):
    """
    Extrait les fronts sur le champ SSH moyen de toute la période.
    Retourne dict { front: array(lon) }.
    """
    print(f"\n[{label}] Extraction sur champ moyen...")
    w    = np.cos(np.deg2rad(ssh_da["lat"]))
    mean_field = ssh_da.mean("time").values   # (lat, lon) — moyenne temporelle simple
    lat  = ssh_da["lat"].values
    lon  = ssh_da["lon"].values
    mean_pos = {}
    for front, level in levels.items():
        mean_pos[front] = _extract_one_field(mean_field, lat, level)
    return mean_pos, lon


# =============================================================================
# 5.  VARIABILITÉ INTERANNUELLE PAR LONGITUDE
# =============================================================================

def compute_variability(ann_pos, label=""):
    """
    Pour chaque front et chaque longitude :
      - std  : écart-type temporel de la latitude annuelle
      - iqr  : P90 - P10
      - trend: tendance linéaire en km/an
      - pval : p-value associée
    Retourne dict { front: dict { 'std', 'iqr', 'trend_km_yr', 'pval',
                                  'mean_lat', 'coverage' } }
    """
    print(f"\n[{label}] Variabilité interannuelle...")
    result = {}
    for front, phi_da in ann_pos.items():
        data  = phi_da.values        # (year, lon)
        years = phi_da["year"].values.astype(float)
        lon   = phi_da["lon"].values
        n_lon = len(lon)

        std_a  = np.full(n_lon, np.nan)
        iqr_a  = np.full(n_lon, np.nan)
        tr_a   = np.full(n_lon, np.nan)
        pv_a   = np.full(n_lon, np.nan)
        cov_a  = np.full(n_lon, np.nan)
        mean_a = np.full(n_lon, np.nan)

        for i in range(n_lon):
            col   = data[:, i]
            valid = np.isfinite(col)
            cov_a[i]  = valid.mean()
            mean_a[i] = np.nanmean(col) if valid.sum() else np.nan
            if valid.sum() < 4:
                continue
            cv, yv    = col[valid], years[valid]
            std_a[i]  = cv.std()
            iqr_a[i]  = np.percentile(cv, 90) - np.percentile(cv, 10)
            sl, _, _, pv, _ = stats.linregress(yv, cv)
            tr_a[i]   = sl * DEG2KM   # km/an
            pv_a[i]   = pv

        def da(arr, nm):
            return xr.DataArray(arr, dims=["lon"], coords={"lon": lon}, name=nm)

        result[front] = {
            "std_km"     : da(std_a * DEG2KM, "std_km"),
            "iqr_km"     : da(iqr_a * DEG2KM, "iqr_km"),
            "trend_km_yr": da(tr_a,            "trend"),
            "pval"       : da(pv_a,            "pval"),
            "mean_lat"   : da(mean_a,           "mean_lat"),
            "coverage"   : da(cov_a,            "coverage"),
        }
        tr_mean = np.nanmean(tr_a)
        pv_mean = np.nanmean(pv_a)
        std_mean = np.nanmean(std_a) * DEG2KM
        print(f"  {front}: std={std_mean:.1f} km  trend={tr_mean:+.2f} km/an "
              f"p={pv_mean:.3f}")
    return result


# =============================================================================
# 6.  FIGURES
# =============================================================================

def _base_map_ax(fig, rect=111):
    """Crée un axe cartopy PlateCarree avec continents, grille et labels lat/lon."""
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -72, -34], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      color="#ccc5ae", zorder=4)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, color="#555", zorder=5)

    gl = ax.gridlines(lw=0.25, color="gray", alpha=0.5,
                      xlocs=range(-180, 181, 30), ylocs=range(-70, -30, 5),
                      draw_labels=True)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#333"}
    gl.ylabel_style = {"size": 8, "color": "#333"}
    ax.set_xlabel("Longitude (°E)", fontsize=9, labelpad=18)
    ax.set_ylabel("Latitude (°N)",  fontsize=9, labelpad=30)
    return ax


def _make_wide_fig():
    """
    Crée une figure large pour les cartes circumpolaires.
    ax.set_aspect('auto') est nécessaire : en PlateCarree, cartopy impose
    sinon le ratio géographique réel (360°lon / 37°lat ≈ 10:1), ce qui
    rend la figure toujours aplatie indépendamment du figsize.
    """
    fig = plt.figure(figsize=(22, 5))
    ax  = _base_map_ax(fig)
    ax.set_aspect("auto")   # laisse matplotlib remplir l'espace disponible
    return fig, ax


def _smooth(arr, w=10):
    """Lissage par moyenne glissante sur w points, gère les NaN."""
    out = np.full_like(arr, np.nan, dtype=float)
    hw  = w // 2
    for i in range(len(arr)):
        seg   = arr[max(0, i - hw): min(len(arr), i + hw + 1)]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= max(2, w // 4):
            out[i] = valid.mean()
    return out


def fig_spaghetti(ann_pos_raw, ann_pos_corr, cm_info, label, outdir):
    """
    Fig 1 : une ligne par année pour chaque front (lon vs lat).
    Produit une version brute et une corrigée si mode commun significatif.
    Les données sont les moyennes annuelles → saisonnalité déjà absente.
    """
    w = CFG["smooth_window"]

    def _make_spaghetti(ann_pos, suffix, title_corr):
        all_years = sorted({int(y) for phi in ann_pos.values()
                            for y in phi["year"].values})
        cmap_t = plt.cm.plasma
        norm_t = mcolors.Normalize(vmin=min(all_years), vmax=max(all_years))

        fig, ax = _make_wide_fig()

        for front in FRONTS:
            phi_da = ann_pos[front]
            lon    = phi_da["lon"].values
            for yr in phi_da["year"].values:
                col    = cmap_t(norm_t(int(yr)))
                phi_sm = _smooth(phi_da.sel(year=yr).values, w)
                ax.plot(lon, phi_sm, color=col, lw=0.7, alpha=0.65,
                        transform=ccrs.PlateCarree(), zorder=3)

            # Étiquette à GAUCHE (premier point valide de la ligne moyenne)
            phi_mean = _smooth(phi_da.mean("year", skipna=True).values, w)
            ok = np.where(np.isfinite(phi_mean))[0]
            if len(ok):
                li = ok[0]   # premier point valide = début (gauche) du front
                ax.text(lon[li] - 2, phi_mean[li], front,
                        transform=ccrs.PlateCarree(), fontsize=8,
                        fontweight="bold", color=FRONT_COLORS[front],
                        ha="right", va="center", zorder=9,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=1))

        # Colorbar années
        sm = plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85, aspect=25)
        cb.set_label("Année", fontsize=9)
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

        trend_str = (f"Mode commun : {cm_info['trend_mm_yr']:+.1f} mm/an  "
                     f"(p={cm_info['pvalue']:.3f})")
        ax.set_title(
            f"Position annuelle des fronts ACC — {label}  [{suffix}]\n"
            f"Moyennes annuelles SSH — saisonnalité implicitement absente\n"
            f"{trend_str}  |  {title_corr}",
            fontsize=10, fontweight="bold", pad=6)

        fname = f"{outdir}fronts_annual_spaghetti_{label}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
        plt.close()
        print(f"  → {fname}")

    _make_spaghetti(ann_pos_raw,  "raw",
                    "Champ SSH brut (mode commun non retiré)")
    if ann_pos_corr is not None:
        _make_spaghetti(ann_pos_corr, "corrected",
                        "Champ SSH corrigé du mode commun")


def fig_variability_map(ann_pos, mean_pos, lon_arr, var_dict, label, outdir):
    """
    Fig 2 : position MOYENNE colorée par variabilité interannuelle (std).
    Dégradé : bleu = stable, rouge = très variable.
    """
    all_std  = np.concatenate([var_dict[f]["std_km"].values for f in FRONTS])
    vmax_std = np.nanpercentile(all_std, 95)
    cmap_v   = plt.cm.RdYlBu_r
    norm_v   = mcolors.Normalize(vmin=0, vmax=vmax_std)
    w = CFG["smooth_window"]

    fig, ax = _make_wide_fig()

    for front in FRONTS:
        phi_mean = _smooth(mean_pos[front], w)
        lon      = lon_arr
        std_km   = var_dict[front]["std_km"].values

        for i in range(len(lon) - 1):
            if not (np.isfinite(phi_mean[i]) and np.isfinite(phi_mean[i + 1])):
                continue
            std_seg = 0.5 * (std_km[i] + std_km[i + 1])
            col_seg = np.nan_to_num(std_seg, nan=0)
            ax.plot([lon[i], lon[i + 1]], [phi_mean[i], phi_mean[i + 1]],
                    color=cmap_v(norm_v(col_seg)), lw=3.5,
                    solid_capstyle="round",
                    transform=ccrs.PlateCarree(), zorder=3)

        # Étiquette à GAUCHE (premier point valide)
        ok = np.where(np.isfinite(phi_mean))[0]
        if len(ok):
            li = ok[0]
            ax.text(lon[li] - 2, phi_mean[li], front,
                    transform=ccrs.PlateCarree(), fontsize=8,
                    fontweight="bold", color="k", ha="right", va="center",
                    zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    sm = plt.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85, aspect=25)
    cb.set_label("Variabilité interannuelle — std (km)", fontsize=9)

    ax.set_title(
        f"Position moyenne des fronts ACC et variabilité interannuelle — {label}\n"
        f"Couleur = std(latitude annuelle) par longitude  "
        f"(bleu = stable, rouge = variable)",
        fontsize=11, fontweight="bold", pad=6)

    fname = f"{outdir}fronts_mean_variability_{label}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


def fig_circumpolar_timeseries(ann_pos_raw, ann_pos_corr, cm_info, label, outdir):
    """
    Fig 3 (diagnostic secondaire) : latitude moyenne circumpolaire par année.
    Montre avant/après correction du mode commun.
    Avertissement : la moyenne circumpolaire peut masquer les contrastes régionaux.
    """
    # Couleurs spécifiques aux séries temporelles
    TS_COLORS = {"SAF": "crimson", "PF": "royalblue", "SACCF": "forestgreen"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Série temporelle circumpolaire moyenne des fronts ACC — {label}\n"
        f"⚠ Diagnostic secondaire : la moyenne circumpolaire peut masquer "
        f"des contrastes régionaux",
        fontsize=10, fontweight="bold")

    for ax, front in zip(axes, FRONTS):
        col = TS_COLORS[front]   # couleur dédiée aux séries temporelles

        for ann_pos, style, lname in [
            (ann_pos_raw,  "-",  "Brut"),
            (ann_pos_corr, "--", "Corrigé mode commun"),
        ]:
            if ann_pos is None:
                continue
            phi_da = ann_pos[front]
            years  = phi_da["year"].values.astype(float)
            lat_cp = np.nanmean(phi_da.values, axis=1)   # moyenne sur longitudes

            ax.plot(years, lat_cp * DEG2KM * 0 + lat_cp,   # valeur en °
                    color=col, ls=style, lw=1.5, alpha=0.9,
                    label=lname, marker="o", ms=3)

            # Tendance
            valid = np.isfinite(lat_cp)
            if valid.sum() > 4:
                sl, it, _, pv, _ = stats.linregress(
                    years[valid], lat_cp[valid])
                trend_km = sl * DEG2KM
                ax.plot(years, it + sl * years, color=col, ls=":",
                        lw=1.0, alpha=0.6)
                txt = (f"{lname}: {trend_km:+.2f} km/an "
                       f"(p={pv:.3f}{'*' if pv < 0.05 else ''})")
                ax.annotate(txt, xy=(0.02, 0.05 if lname == "Brut" else 0.18),
                            xycoords="axes fraction", fontsize=7,
                            color=col, alpha=0.9)

        ax.invert_yaxis()
        ax.set_ylabel("Lat. moy. (°)", fontsize=9)
        ax.set_title(front, fontsize=10, fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Année", fontsize=9)
    fname = f"{outdir}fronts_circumpolar_timeseries_{label}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# =============================================================================
# 7.  RÉSUMÉ CSV + CONSOLE
# =============================================================================

def export_summary(var_dict, cm_info, ann_pos, label, outdir):
    """
    Exporte un CSV récapitulatif et affiche les chiffres clés.
    """
    rows = []
    for front in FRONTS:
        v   = var_dict[front]
        phi = ann_pos[front]

        lat_mean = float(v["mean_lat"].mean(skipna=True))
        std_mean = float(v["std_km"].mean(skipna=True))
        iqr_mean = float(v["iqr_km"].mean(skipna=True))
        cov_mean = float(v["coverage"].mean(skipna=True)) * 100
        tr_mean  = float(v["trend_km_yr"].mean(skipna=True))
        pv_mean  = float(v["pval"].mean(skipna=True))

        rows.append({
            "Dataset"                     : label,
            "Front"                       : front,
            "SSH_level_m"                 : CFG[f"levels_{'obs' if 'OBS' in label else 'nemo'}"][front],
            "Lat_moy_circ_deg"            : f"{lat_mean:.2f}",
            "Std_interannuel_km"          : f"{std_mean:.1f}",
            "IQR_P90_P10_km"             : f"{iqr_mean:.1f}",
            "Couverture_longitudinale_pct": f"{cov_mean:.1f}",
            "Trend_km_par_an"             : f"{tr_mean:+.3f}",
            "Trend_km_par_decennie"       : f"{tr_mean*10:+.2f}",
            "Pval_trend"                  : f"{pv_mean:.4f}",
            "Significatif_5pct"           : "oui" if pv_mean < 0.05 else "non",
            "Mode_commun_mm_par_an"       : f"{cm_info['trend_mm_yr']:+.2f}",
            "Mode_commun_pval"            : f"{cm_info['pvalue']:.4f}",
            "Mode_commun_significatif"    : "oui" if cm_info["significant"] else "non",
            "Methode_correction"          : (
                "mode commun soustrait" if cm_info["significant"]
                else "aucune correction (non significatif)"),
            "Note_NEMO"                   : (
                "tendance large échelle SSH (piControl absent, dérive non prouvée)"
                if "NEMO" in label else "N/A"),
        })

    df = pd.DataFrame(rows)
    fname_csv = f"{outdir}summary_{label}.csv"
    df.to_csv(fname_csv, index=False, sep=";")
    print(f"  → {fname_csv}")

    # Affichage console formaté
    print(f"\n{'='*62}")
    print(f"  RÉSUMÉ — {label}")
    print(f"{'='*62}")
    print(f"\n  MODE COMMUN SSH ({label})")
    print(f"    Tendance  : {cm_info['trend_mm_yr']:+.2f} mm/an = "
          f"{cm_info['trend_cm_dec']:+.2f} cm/décennie")
    print(f"    R²        : {cm_info['r2']:.3f}   p = {cm_info['pvalue']:.4f}   "
          f"{'✓ SIGNIFICATIF' if cm_info['significant'] else '✗ non sig.'}")
    if "NEMO" in label:
        print(f"    ⚠ Nommé 'tendance large échelle SSH' (pas de piControl disponible)")

    print(f"\n  VARIABILITÉ INTERANNUELLE DES FRONTS")
    print(f"  {'Front':<8} {'Lat moy':>8} {'Std':>8} {'IQR':>8} "
          f"{'Trend':>12} {'p':>8}")
    print(f"  {'':8} {'(°)':>8} {'(km)':>8} {'(km)':>8} "
          f"{'(km/an)':>12} {'':>8}")
    print(f"  {'-'*56}")
    for front in FRONTS:
        v   = var_dict[front]
        sig = "(*)" if float(v["pval"].mean(skipna=True)) < 0.05 else "   "
        print(f"  {front:<8} "
              f"{float(v['mean_lat'].mean(skipna=True)):>8.2f} "
              f"{float(v['std_km'].mean(skipna=True)):>8.1f} "
              f"{float(v['iqr_km'].mean(skipna=True)):>8.1f} "
              f"{float(v['trend_km_yr'].mean(skipna=True)):>+12.3f} "
              f"{float(v['pval'].mean(skipna=True)):>8.4f} {sig}")
    print(f"{'='*62}\n")

    return df


# =============================================================================
# 8.  PIPELINE PRINCIPAL
# =============================================================================
def export_longitudinal_profiles(var_dict, label, outdir):
    """Exporte std, trend et pval par longitude pour chaque front."""
    frames = []
    for front in FRONTS:
        v   = var_dict[front]
        lon = v["std_km"]["lon"].values
        df  = pd.DataFrame({
            "lon"          : lon,
            "front"        : front,
            "std_km"       : v["std_km"].values,
            "iqr_km"       : v["iqr_km"].values,
            "trend_km_yr"  : v["trend_km_yr"].values,
            "trend_km_dec" : v["trend_km_yr"].values * 10,
            "pval"         : v["pval"].values,
            "signif_5pct"  : (v["pval"].values < 0.05).astype(int),
            "mean_lat_deg" : v["mean_lat"].values,
        })
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    fname = f"{outdir}longitudinal_profiles_{label}.csv"
    out.to_csv(fname, index=False, sep=";")
    print(f"  → {fname}")
    return out

def process_dataset(ssh_da, levels, label):
    """
    Traite un jeu de données complet (OBS ou NEMO) :
      1. Mode commun
      2. Positions annuelles brutes et corrigées
      3. Variabilité
      4. Figures
      5. CSV
    """
    outdir = CFG["outdir"]

    # 1. Mode commun
    cm_da, cm_info = detect_common_mode(ssh_da, label)

    # 2. Positions annuelles — version brute
    print(f"\n--- {label} : positions annuelles (version brute) ---")
    ann_raw  = extract_annual_positions(ssh_da, levels, label + "_raw")
    mean_raw, lon_arr = extract_mean_position(ssh_da, levels, label + "_raw")

    # 3. Positions annuelles — version corrigée
    ssh_corr = apply_common_mode_correction(ssh_da, cm_da)
    print(f"\n--- {label} : positions annuelles (mode commun corrigé) ---")
    ann_corr  = extract_annual_positions(ssh_corr, levels, label + "_corr")
    mean_corr, _ = extract_mean_position(ssh_corr, levels, label + "_corr")

    # 4. Variabilité (sur version corrigée = plus propre)
    var_dict = compute_variability(ann_corr, label)

    # 5. Figures
    print(f"\n--- {label} : figures ---")
    fig_spaghetti(ann_raw, ann_corr, cm_info, label, outdir)
    fig_variability_map(ann_corr, mean_corr, lon_arr, var_dict, label, outdir)
    fig_circumpolar_timeseries(ann_raw, ann_corr, cm_info, label, outdir)

    # 6. CSV + console
    df = export_summary(var_dict, cm_info, ann_corr, label, outdir)
    export_longitudinal_profiles(var_dict, label, outdir)
    ds_ann = xr.Dataset({f: ann_corr[f] for f in FRONTS})
    ds_ann.to_netcdf(f"{outdir}ann_pos_corr_{label}.nc")
    export_longitudinal_profiles(var_dict, label, outdir)
    return {
        "ssh"      : ssh_da,
        "ann_raw"  : ann_raw,
        "ann_corr" : ann_corr,
        "mean_corr": mean_corr,
        "var_dict" : var_dict,
        "cm_info"  : cm_info,
        "summary"  : df,
    }


def run():
    os.makedirs(CFG["outdir"], exist_ok=True)

    print("=" * 62)
    print("  ANALYSE VARIABILITÉ FRONTS ACC")
    print(f"  Fronts : {FRONTS}")
    print(f"  Sorties : {CFG['outdir']}")
    print("=" * 62)

    # ─── OBS ──────────────────────────────────────────────────────────────
    ssh_obs = load_obs(CFG)
    res_obs = process_dataset(ssh_obs, CFG["levels_obs"], "OBS")

    # ─── NEMO ─────────────────────────────────────────────────────────────
    ssh_nemo = load_nemo(CFG)
    res_nemo = process_dataset(ssh_nemo, CFG["levels_nemo"], "NEMO")

    print("\n  Fichiers produits :")
    for f in sorted(os.listdir(CFG["outdir"])):
        print(f"    {CFG['outdir']}{f}")

    return {"obs": res_obs, "nemo": res_nemo}


if __name__ == "__main__":
    results = run()