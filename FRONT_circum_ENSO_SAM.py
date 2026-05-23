"""
=============================================================================
FRONTS ACC — SÉRIE TEMPORELLE CIRCUMPOLAIRE AVEC PHASES ENSO/SAM
+ PROFILS LONGITUDINAUX PAR PHASE (vue régionale)
=============================================================================
Entrées :
  outputs_position/ann_pos_corr_OBS.nc
  outputs_position/ann_pos_corr_NEMO.nc

Sorties dans ./outputs_enso_sam/ :
  timeseries_phases_OBS.png   — même que fronts_circumpolar_timeseries_OBS.png
  timeseries_phases_NEMO.png    mais chaque point coloré = phase ENSO,
                                forme du marqueur = phase SAM
  longitudinal_phases_OBS.png — profil lon×lat par phase (vue régionale)
  longitudinal_phases_NEMO.png
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PARAMÈTRES
# =============================================================================
FILES = {
    "OBS" : "outputs_position/ann_pos_corr_OBS.nc",
    "NEMO": "outputs_position/ann_pos_corr_NEMO.nc",
}
OUTDIR  = "outputs_enso_sam/"
DPI     = 150
DEG2KM  = 111.0
FRONTS  = ["SAF", "PF", "SACCF"]
SMOOTH_W = 10   # lissage longitudinal pour la vue régionale

# =============================================================================
# TABLE DES PHASES  (décembre–mars, d'après le tableau fourni)
# =============================================================================
PHASES = pd.DataFrame([
    (1991, "Neutre",   "Neutre"),
    (1992, "El Niño",  "Negative"),
    (1993, "Neutre",   "Negative"),
    (1994, "Neutre",   "Positive"),
    (1995, "El Niño",  "Positive"),
    (1996, "La Niña",  "Neutre"),
    (1997, "Neutre",   "Neutre"),
    (1998, "El Niño",  "Neutre"),
    (1999, "La Niña",  "Positive"),
    (2000, "La Niña",  "Positive"),
    (2001, "La Niña",  "Negative"),
    (2002, "Neutre",   "Neutre"),
    (2003, "El Niño",  "Neutre"),
    (2004, "Neutre",   "Negative"),
    (2005, "El Niño",  "Neutre"),
    (2006, "La Niña",  "Negative"),
    (2007, "Neutre",   "Neutre"),
    (2008, "La Niña",  "Positive"),
    (2009, "La Niña",  "Positive"),
    (2010, "El Niño",  "Neutre"),
    (2011, "La Niña",  "Neutre"),
    (2012, "La Niña",  "Positive"),
    (2013, "Neutre",   "Neutre"),
    (2014, "Neutre",   "Neutre"),
    (2015, "El Niño",  "Positive"),
    (2016, "El Niño",  "Positive"),
    (2017, "Neutre",   "Negative"),
    (2018, "La Niña",  "Positive"),
    (2019, "El Niño",  "Neutre"),
    (2020, "El Niño",  "Neutre"),
], columns=["year", "ENSO", "SAM"])
PHASES = PHASES.set_index("year")

# Couleurs ENSO (fond du marqueur)
ENSO_COLOR = {"El Niño": "#d73027", "La Niña": "#4575b4", "Neutre": "#888888"}
# Formes SAM (forme du marqueur)
SAM_MARKER = {"Positive": "^", "Neutre": "o", "Negative": "v"}
# Couleurs des courbes composites par phase
ENSO_LINE  = {"El Niño": "#d73027", "La Niña": "#4575b4", "Neutre": "#888888"}
SAM_LINE   = {"Positive": "#1b7837", "Neutre":  "#888888",  "Negative": "#762a83"}

ENSO_ORDER = ["El Niño", "Neutre", "La Niña"]
SAM_ORDER  = ["Positive", "Neutre", "Negative"]


# =============================================================================
# UTILITAIRES
# =============================================================================

def circumpolar_mean(ds):
    """Latitude circumpolaire moyenne (moyenne sur lon) pour chaque année."""
    years = ds.year.values.astype(int)
    data  = {}
    for f in FRONTS:
        if f in ds:
            data[f] = np.nanmean(ds[f].values, axis=1)   # (year,)
    return pd.DataFrame(data, index=years)


def smooth(arr, w=10):
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        seg = arr[max(0, i - w//2): min(len(arr), i + w//2 + 1)]
        v   = seg[np.isfinite(seg)]
        if len(v) >= max(2, w // 4):
            out[i] = v.mean()
    return out


# =============================================================================
# FIGURE 1 — Série temporelle circumpolaire annotée (reproduit timeseries v6)
# =============================================================================

def fig_timeseries(nc_path, label):
    """
    3 sous-graphes (SAF / PF / SACCF).
    Ligne grise = tendance linéaire.
    Chaque année = un marqueur :
      couleur  → phase ENSO   (rouge = El Niño, bleu = La Niña, gris = Neutre)
      forme    → phase SAM    (▲ = Positive, ● = Neutre, ▼ = Negative)
    Annotation de l'année à côté de chaque point.
    """
    ds     = xr.open_dataset(nc_path)
    circ   = circumpolar_mean(ds)
    fronts = [f for f in FRONTS if f in circ.columns]

    # Intersection années communes
    common = sorted(set(circ.index) & set(PHASES.index))
    circ   = circ.loc[common]
    ph     = PHASES.loc[common]

    nf  = len(fronts)
    fig, axes = plt.subplots(nf, 1, figsize=(14, 4 * nf), sharex=True)
    if nf == 1:
        axes = [axes]

    fig.suptitle(
        f"Position circumpolaire des fronts ACC — {label}\n"
        f"Couleur = phase ENSO  |  Forme = phase SAM",
        fontsize=12, fontweight="bold")

    years = np.array(common, dtype=float)

    for ax, front in zip(axes, fronts):
        lat = circ[front].values

        # Trait de fond (ligne reliant les points)
        ax.plot(years, lat, color="#cccccc", lw=1.0, zorder=1)

        # Tendance linéaire
        valid = np.isfinite(lat)
        if valid.sum() > 4:
            sl, it, _, pv, _ = stats.linregress(years[valid], lat[valid])
            ax.plot(years, it + sl * years, color="k", ls="--", lw=1.2,
                    alpha=0.6, zorder=2)
            sig = "*" if pv < 0.05 else ""
            ax.text(years[-1] + 0.3, it + sl * years[-1],
                    f"{sl * DEG2KM:+.2f} km/an{sig}",
                    fontsize=8, va="center", color="k")

        # Marqueurs annotés
        for yr_i, yr in enumerate(common):
            enso = ph.loc[yr, "ENSO"]
            sam  = ph.loc[yr, "SAM"]
            y_val = lat[yr_i]
            if not np.isfinite(y_val):
                continue
            ax.scatter(yr, y_val,
                       color=ENSO_COLOR[enso],
                       marker=SAM_MARKER[sam],
                       s=90, zorder=5,
                       edgecolors="k", linewidths=0.5)
            # Annotation de l'année (en petit, légèrement décalée)
            ax.annotate(str(yr),
                        xy=(yr, y_val), xytext=(0, 6),
                        textcoords="offset points",
                        fontsize=6, ha="center", color="#333",
                        zorder=6)

        ax.invert_yaxis()
        ax.set_ylabel(f"Lat. circumpolaire (°)", fontsize=9)
        ax.set_title(front, fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(years[0] - 1, years[-1] + 2)

    axes[-1].set_xlabel("Année", fontsize=10)

    # ── Légendes ──────────────────────────────────────────────────────────
    enso_handles = [
        mlines.Line2D([], [], color=ENSO_COLOR[p], marker="o", ls="None",
                      ms=9, markeredgecolor="k", markeredgewidth=0.5,
                      label=f"ENSO : {p}")
        for p in ENSO_ORDER
    ]
    sam_handles = [
        mlines.Line2D([], [], color="gray", marker=SAM_MARKER[p], ls="None",
                      ms=9, markeredgecolor="k", markeredgewidth=0.5,
                      label=f"SAM : {p}")
        for p in SAM_ORDER
    ]
    axes[0].legend(handles=enso_handles + sam_handles,
                   fontsize=8, loc="upper right", ncol=6,
                   framealpha=0.9)

    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fname = f"{OUTDIR}timeseries_phases_{label}.png"
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# =============================================================================
# FIGURE 2 — Profils longitudinaux par phase (ENSO et SAM séparément)
# Répond à la question : influence homogène circumpolaire ou régionale ?
# =============================================================================

def fig_longitudinal(nc_path, label):
    """
    Pour chaque front : 2 panneaux côte-à-côte
      Gauche  : composite latitude(lon) par phase ENSO
      Droite  : composite latitude(lon) par phase SAM
    X = longitude (−180 → 180), Y = latitude du front.
    Enveloppe ±σ = dispersion interannuelle au sein de chaque phase.
    → courbes parallèles  → influence homogène circumpolaire
    → courbes qui divergent par secteur → influence régionale
    """
    ds       = xr.open_dataset(nc_path)
    lon      = ds.lon.values
    fronts   = [f for f in FRONTS if f in ds]
    years_nc = ds.year.values.astype(int)
    common   = sorted(set(years_nc) & set(PHASES.index))
    ph       = PHASES.loc[common]

    # Secteurs géographiques (bandes alternées de fond)
    SECTORS = [
        (-180, -70, "Pac. E.",  "#f5f5f5"),
        ( -70,  20, "Atlantique","#eaf0fb"),
        (  20,  90, "Indien",   "#f5f5f5"),
        (  90, 180, "Pac. O.",  "#eaf0fb"),
    ]

    nf  = len(fronts)
    fig, axes = plt.subplots(nf, 2, figsize=(20, 4.5 * nf),
                             sharey="row", sharex="col")
    if nf == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Position longitudinale des fronts ACC par phase — {label}\n"
        f"Trait plein = moyenne de phase  |  Enveloppe = ±1σ interannuel  |  "
        f"Pointillé noir = moyenne toutes années",
        fontsize=11, fontweight="bold")

    for row, front in enumerate(fronts):
        phi_all    = ds[front].sel(year=common).values          # (year, lon)
        phi_global = smooth(np.nanmean(phi_all, axis=0), SMOOTH_W)

        for col, (phase_var, phase_order, pal) in enumerate([
            ("ENSO", ENSO_ORDER, ENSO_LINE),
            ("SAM",  SAM_ORDER,  SAM_LINE),
        ]):
            ax = axes[row, col]

            # 1. Bandes de fond secteurs
            for x0, x1, sname, scol in SECTORS:
                ax.axvspan(x0, x1, color=scol, alpha=0.7, zorder=0)
                ax.text((x0 + x1) / 2, 0.98, sname,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7,
                        color="#aaaaaa", style="italic")

            # 2. Moyenne globale de référence
            ax.plot(lon, phi_global, color="k", lw=1.8, ls="--",
                    alpha=0.55, label="Toutes années", zorder=5)

            # 3. Composite par phase
            for phase in phase_order:
                yrs_ph = [y for y in common if ph.loc[y, phase_var] == phase]
                if not yrs_ph:
                    continue
                idx   = [list(common).index(y) for y in yrs_ph]
                phi_ph = phi_all[idx, :]
                m_ph   = smooth(np.nanmean(phi_ph, axis=0), SMOOTH_W)
                s_ph   = np.nanstd(phi_ph, axis=0)

                ax.plot(lon, m_ph, color=pal[phase], lw=2.2,
                        label=f"{phase} (n={len(yrs_ph)})", zorder=6)
                ax.fill_between(lon, m_ph - s_ph, m_ph + s_ph,
                                color=pal[phase], alpha=0.15, zorder=3)

            # 4. Mise en forme — APRÈS avoir tracé les données
            ax.invert_yaxis()
            ax.set_xlim(-180, 180)
            ax.set_xticks(range(-180, 181, 30))
            ax.set_xlabel("Longitude (°)", fontsize=9)
            ax.set_ylabel("Latitude (°)", fontsize=9)
            ax.set_title(f"{front} — phases {phase_var}", fontsize=10,
                         fontweight="bold")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                      ncol=len(phase_order) + 1)
            ax.grid(True, axis="y", alpha=0.3, lw=0.6)
            ax.grid(True, axis="x", alpha=0.2, lw=0.4)

    plt.tight_layout()
    fname = f"{OUTDIR}longitudinal_phases_{label}.png"
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# =============================================================================
# MAIN
# =============================================================================

def run():
    os.makedirs(OUTDIR, exist_ok=True)
    for label, path in FILES.items():
        if not os.path.exists(path):
            print(f"⚠  {path} introuvable — ignoré")
            continue
        print(f"\n{'='*55}")
        print(f"  {label}  ({path})")
        print(f"{'='*55}")
        fig_timeseries(path, label)
        fig_longitudinal(path, label)

    print(f"\nFichiers produits dans {OUTDIR} :")
    for f in sorted(os.listdir(OUTDIR)):
        print(f"  {f}")


if __name__ == "__main__":
    run()