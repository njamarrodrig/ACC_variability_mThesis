"""
=============================================================================
FRONT_REGIONAL_V2.py — Variabilité régionale des fronts ACC
=============================================================================
Exploite les sorties de FRONT_VARIABILITY_V6.py (après ajout des lignes
de sauvegarde dans process_dataset) :
  outputs_position/ann_pos_corr_OBS.nc
  outputs_position/ann_pos_corr_NEMO.nc
  outputs_position/longitudinal_profiles_OBS.csv
  outputs_position/longitudinal_profiles_NEMO.csv

Workflow :
  1. Charger les profils σ_f(λ) et afficher les zones sélectionnées
     → region_selection_OBS.png / _NEMO.png  (vérification visuelle)
  2. Pour chaque zone : série temporelle régionale, diagnostics, figure
     → regional_<zone>.png
  3. Figure de synthèse toutes zones
     → regional_summary_all_zones.png
  4. CSV récapitulatif
     → regional_summary.csv

SAM / ENSO : délibérément absents — traités dans la discussion.

Comparaison OBS / NEMO :
  Biais de position, amplitude relative (σ_NEMO/σ_OBS), covariabilité.
  La variabilité NEMO peut apparaître 2–3× plus élevée que les OBS.
  Causes probables : résolution plus fine (NEMO ~1/4° vs OBS ~1°),
  période plus longue (33 ans vs 17 ans), absence de lissage instrumental.
  Le ratio σ_NEMO/σ_OBS est donc reporté explicitement, sans en conclure
  à un excès de variabilité dynamique.

=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION DES ZONES
# =============================================================================
# Définir les bornes à partir de la figure fronts_mean_variability_*.png.
# Critère suggéré : σ_f(λ) > moyenne circumpolaire + 0.5 × σ_global.
# Ajuster selon la topographie (ride de mi-océan, passage, plateau...).
# Laisser "fronts" vide pour traiter les 3 fronts automatiquement.

ZONES = {
    # Variabilité libre du bassin Pacifique Sud, avant la UFZ
    "PacSud"      : {"lon": (-175, -155), "topo": "Pacifique Sud — variabilité de bassin"},

    # UFZ — zone à part, interprétation topographique uniquement
    "UFZ"         : {"lon": (-152, -143), "topo": "Fracture Udintsev — épinglage topographique"},

    # OBS seulement, dorsale Est-Pacifique
    "EPacifique"  : {"lon": (-130, -110), "topo": "Dorsale Est-Pacifique (signal OBS)"},

    "Drake"       : {"lon": (-70,  -50),  "topo": "Passage de Drake"},
    "AtlSudOuest" : {"lon": (-50,  -15),  "topo": "Aval dorsale Scotia"},
    "Kerguelen"   : {"lon": ( 60,   90),  "topo": "Plateau de Kerguelen"},
    "Campbell"    : {"lon": (155,  175),  "topo": "Ridge de Campbell / NZ"},
}

FRONTS       = ["SAF", "PF", "SACCF"]
FRONT_COLORS = {"SAF": "#d6604d", "PF": "#2166ac", "SACCF": "#4dac26"}
DEG2KM       = 111.0

CFG = {
    "outdir"         : "outputs_regional/",
    "ann_obs_file"   : "outputs_position/ann_pos_corr_OBS.nc",
    "ann_nemo_file"  : "outputs_position/ann_pos_corr_NEMO.nc",
    "profiles_obs"   : "outputs_position/longitudinal_profiles_OBS.csv",
    "profiles_nemo"  : "outputs_position/longitudinal_profiles_NEMO.csv",
    "common_period"  : (2002, 2018),   # période commune OBS / NEMO
    "fig_dpi"        : 150,
    "pval_threshold" : 0.05,
    # Note résolution — affichée dans les figures de comparaison
    "res_note"       : ("σ_NEMO/σ_OBS peut atteindre 2–3 : résolution ~1/4° (NEMO) "
                        "vs ~1° (OBS), période +16 ans, pas de lissage instrumental."),
}


# =============================================================================
# 1.  CHARGEMENT
# =============================================================================

def load_annual_positions(nc_file, label):
    ds  = xr.open_dataset(nc_file)
    pos = {f: ds[f] for f in FRONTS if f in ds}
    print(f"[{label}] Positions chargées : {list(pos.keys())} | "
          f"années {int(ds.year.values[0])}–{int(ds.year.values[-1])}")
    return pos


def load_profiles(csv_file, label):
    df = pd.read_csv(csv_file, sep=";")
    print(f"[{label}] Profils longitudinaux : {len(df)} lignes")
    return df


# =============================================================================
# 2.  AIDE À LA SÉLECTION — profils σ avec zones surlignées
# =============================================================================

def suggest_regions(profiles_df, label, factor=0.5):
    """
    Affiche dans le terminal les longitudes où σ_f(λ) > μ + factor×σ_global.
    Aide à définir ou ajuster les bornes des ZONES.
    """
    print(f"\n[{label}] Suggestions (σ > μ+{factor}σ) :")
    for front in FRONTS:
        sub   = profiles_df[profiles_df["front"] == front]
        mu    = sub["std_km"].mean()
        sg    = sub["std_km"].std()
        above = sub[sub["std_km"] > mu + factor * sg]["lon"].values
        if len(above) == 0:
            print(f"  {front}: aucune zone")
            continue
        gaps = np.where(np.diff(above) > 5)[0]
        segs = np.split(above, gaps + 1)
        print(f"  {front} (seuil={mu+factor*sg:.0f} km) :")
        for seg in [s for s in segs if len(s) >= 3]:
            print(f"    λ=[{seg[0]:.0f}°, {seg[-1]:.0f}°]  "
                  f"σ_max={sub[sub['lon'].isin(seg)]['std_km'].max():.0f} km")


def fig_region_selection(profiles_df, zones, label, outdir):
    """
    Profils σ_f(λ) avec zones surlignées — permet de vérifier visuellement
    que les bornes sont bien positionnées avant de lancer l'analyse.
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    fig.suptitle(
        f"Profils σ_f(λ) et zones sélectionnées — {label}\n"
        f"Zones en jaune = régions retenues pour l'analyse régionale",
        fontsize=15, fontweight="bold")

    for ax, front in zip(axes, FRONTS):
        sub = profiles_df[profiles_df["front"] == front]
        col = FRONT_COLORS[front]

        ax.fill_between(sub["lon"], 0, sub["std_km"], color=col, alpha=0.15)
        ax.plot(sub["lon"], sub["std_km"], color=col, lw=1.5, label=front)

        # Seuil de référence
        mu, sg = sub["std_km"].mean(), sub["std_km"].std()
        ax.axhline(mu + 0.5 * sg, color="k", ls="--", lw=0.8,
                   label=f"μ+0.5σ = {mu+0.5*sg:.0f} km")

        # Zones
        y_max = sub["std_km"].max() * 1.05
        for zname, zcfg in zones.items():
            l1, l2 = zcfg["lon"]
            ax.axvspan(l1, l2, color="gold", alpha=0.30, zorder=0)
            ax.text((l1 + l2) / 2, y_max, zname,
                    ha="center", va="top", fontsize=15, color="#555",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

        ax.set_ylabel("σ (km)", fontsize=15)
        ax.set_title(front, fontsize=17, fontweight="bold", loc="left")
        ax.legend(fontsize=15, loc="upper right")
        ax.tick_params(labelsize=15)
        ax.set_xlim(-180, 180)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Longitude (°E)", fontsize=17)
    fname = f"{outdir}region_selection_{label}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# =============================================================================
# 3.  SÉRIE TEMPORELLE RÉGIONALE
# =============================================================================

def build_regional_ts(ann_pos, lon1, lon2):
    """
    φ̃_f(Y) = moyenne spatiale de φ_f(λ,Y) sur [lon1, lon2].
    Retourne dict {front: {'years', 'phi_deg', 'anom_km', 'phi_mean_deg',
                            'coverage', 'n_lon'}}
    """
    result = {}
    for front in FRONTS:
        if front not in ann_pos:
            continue
        phi_da   = ann_pos[front]
        lon      = phi_da["lon"].values
        years    = phi_da["year"].values.astype(int)
        mask_lon = (lon >= lon1) & (lon <= lon2)

        if mask_lon.sum() == 0:
            print(f"  ⚠ {front} : aucune longitude dans [{lon1}°,{lon2}°]")
            continue

        phi_zone = phi_da.values[:, mask_lon]          # (year, lon_zone)
        n_valid  = np.isfinite(phi_zone).sum(axis=1)   # (year,)
        phi_mean_yr = np.where(n_valid > 0,
                               np.nanmean(phi_zone, axis=1), np.nan)
        coverage = n_valid / mask_lon.sum()

        phi_mean_all = np.nanmean(phi_mean_yr[np.isfinite(phi_mean_yr)])
        anom_km      = (phi_mean_yr - phi_mean_all) * DEG2KM

        result[front] = {
            "years"        : years,
            "phi_deg"      : phi_mean_yr,
            "anom_km"      : anom_km,
            "phi_mean_deg" : phi_mean_all,
            "coverage"     : coverage,
            "n_lon"        : int(mask_lon.sum()),
        }
    return result


# =============================================================================
# 4.  DIAGNOSTICS
# =============================================================================

def compute_diagnostics(ts_dict, label=""):
    """
    σ_f (km), amplitude totale A_f (km), tendance linéaire (km/déc), p-value.
    """
    diags = {}
    for front, ts in ts_dict.items():
        years = ts["years"].astype(float)
        phi   = ts["phi_deg"]
        valid = np.isfinite(phi)

        if valid.sum() < 4:
            print(f"  ⚠ [{label}] {front} : {valid.sum()} pts valides — ignoré")
            continue

        yv, pv    = years[valid], phi[valid]
        sigma_km  = pv.std(ddof=1) * DEG2KM
        amp_km    = (pv.max() - pv.min()) * DEG2KM

        sl, it, r, pval, _ = stats.linregress(yv, pv)
        trend_km_dec = sl * DEG2KM * 10

        diags[front] = {
            "years"        : years,
            "phi_deg"      : phi,
            "anom_km"      : ts["anom_km"],
            "phi_mean_deg" : ts["phi_mean_deg"],
            "sigma_km"     : sigma_km,
            "amp_km"       : amp_km,
            "slope_deg_yr" : sl,
            "intercept"    : it,
            "trend_km_dec" : trend_km_dec,
            "pval"         : pval,
            "r2"           : r**2,
            "n_valid"      : int(valid.sum()),
            "n_lon"        : ts["n_lon"],
        }
    return diags


# =============================================================================
# 5.  COMPARAISON OBS / NEMO SUR PÉRIODE COMMUNE
# =============================================================================

def compare_obs_nemo(diags_obs, diags_nemo):
    """
    Sur la période commune (CFG["common_period"]) :
      - biais de position moyenne (NEMO − OBS) en km
      - ratio σ_NEMO / σ_OBS  (attendu > 1 pour des raisons de résolution)
      - corrélation interannuelle des anomalies (covariabilité)
    Retourne dict {front: dict | None}
    """
    y1, y2  = CFG["common_period"]
    result  = {}

    for front in FRONTS:
        if front not in diags_obs or front not in diags_nemo:
            result[front] = None
            continue

        do, dn = diags_obs[front], diags_nemo[front]

        # Années communes dans la période
        yr_common = np.intersect1d(
            do["years"][(do["years"] >= y1) & (do["years"] <= y2)],
            dn["years"][(dn["years"] >= y1) & (dn["years"] <= y2)])

        if len(yr_common) < 4:
            result[front] = None
            continue

        def _pick(d, yrs):
            return np.array([d["phi_deg"][d["years"] == y][0] for y in yrs])

        po = _pick(do, yr_common)
        pn = _pick(dn, yr_common)
        ok = np.isfinite(po) & np.isfinite(pn)
        if ok.sum() < 4:
            result[front] = None
            continue

        po_v, pn_v = po[ok], pn[ok]
        bias_km    = (np.mean(pn_v) - np.mean(po_v)) * DEG2KM
        std_obs_km = np.std(po_v, ddof=1) * DEG2KM
        std_nmo_km = np.std(pn_v, ddof=1) * DEG2KM
        ratio_std  = std_nmo_km / std_obs_km if std_obs_km > 0 else np.nan
        r_cov, p_cov = stats.pearsonr(po_v - po_v.mean(),
                                       pn_v - pn_v.mean())

        result[front] = {
            "n_common"    : int(ok.sum()),
            "yr_common"   : yr_common[ok],
            "bias_km"     : bias_km,
            "std_obs_km"  : std_obs_km,
            "std_nmo_km"  : std_nmo_km,
            "ratio_std"   : ratio_std,
            "r_cov"       : r_cov,
            "p_cov"       : p_cov,
            "po_v"        : po_v,
            "pn_v"        : pn_v,
        }

    return result


# =============================================================================
# 6.  FIGURE PAR ZONE
# =============================================================================
def fig_zone(zname, zcfg, diags_obs, diags_nemo, comp_dict, outdir):
    """
    Pour une zone : 3 panneaux (un par front), sans tableau.
    Axe principal = anomalie de position en km (centrée sur la moyenne OBS).
    """
    lon1, lon2 = zcfg["lon"]
    y1_cp, y2_cp = CFG["common_period"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.patch.set_facecolor("#f9f9f9")
    fig.suptitle(
        f"Variabilité régionale des fronts — {zname}  "
        f"[{lon1}°–{lon2}°E]\n{zcfg['topo']}",
        fontsize=12, fontweight="bold", y=1.00)

    for i_f, front in enumerate(FRONTS):
        ax  = axes[i_f]
        col = FRONT_COLORS[front]

        # ── OBS ──────────────────────────────────────────────────────────
        if front in diags_obs:
            do    = diags_obs[front]
            years = do["years"]
            anom  = do["anom_km"]
            valid = np.isfinite(anom)

            ax.plot(years[valid], anom[valid],
                    color=col, lw=2.0, ls="-", marker="o", ms=4.5,
                    label="OBS", zorder=4)

            sig = "*" if do["pval"] < CFG["pval_threshold"] else " (ns)"
            t_anom = (do["slope_deg_yr"] * do["years"] + do["intercept"]
                      - do["phi_mean_deg"]) * DEG2KM
            ax.plot(years[valid], t_anom[valid],
                    color=col, lw=1.0, ls=":", alpha=0.7, zorder=3)
            ax.annotate(
                f"OBS: {do['trend_km_dec']:+.1f} km/déc{sig}  "
                f"σ={do['sigma_km']:.0f} km  n={do['n_valid']} ans",
                xy=(0.02, 0.10), xycoords="axes fraction",
                fontsize=8, color=col)

        # ── NEMO ─────────────────────────────────────────────────────────
        if front in diags_nemo:
            dn    = diags_nemo[front]
            years = dn["years"]
            ref_obs = diags_obs[front]["phi_mean_deg"] if front in diags_obs else dn["phi_mean_deg"]
            anom_n  = (dn["phi_deg"] - ref_obs) * DEG2KM
            valid   = np.isfinite(anom_n)

            ax.plot(years[valid], anom_n[valid],
                    color=col, lw=1.4, ls="--", marker="s", ms=3.5,
                    alpha=0.75, label="NEMO", zorder=3)

            sig = "*" if dn["pval"] < CFG["pval_threshold"] else " (ns)"
            t_anom_n = (dn["slope_deg_yr"] * dn["years"] + dn["intercept"]
                        - ref_obs) * DEG2KM
            ax.plot(years[valid], t_anom_n[valid],
                    color=col, lw=0.8, ls=(0, (4, 2, 1, 2)),
                    alpha=0.55, zorder=2)
            ax.annotate(
                f"NEMO: {dn['trend_km_dec']:+.1f} km/déc{sig}  "
                f"σ={dn['sigma_km']:.0f} km  n={dn['n_valid']} ans",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color=col, alpha=0.85)

        # ── Zone commune + mise en forme ──────────────────────────────────
        ax.axvspan(y1_cp, y2_cp, color="lightgrey", alpha=0.35, zorder=0,
                   label=f"Période commune ({y1_cp}–{y2_cp})")
        ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
        ax.set_ylabel("Anomalie (km)\n[+ = vers le Nord]", fontsize=11)
        ax.set_title(front, fontsize=10, fontweight="bold",
                     color=col, loc="left")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.25)

        # Comparaison OBS↔NEMO sur la période commune
        c = comp_dict.get(front)
        if c is not None:
            yc  = c["yr_common"].astype(float)
            po  = (c["po_v"] - diags_obs[front]["phi_mean_deg"]) * DEG2KM
            pn  = (c["pn_v"] - diags_obs[front]["phi_mean_deg"]) * DEG2KM
            ax.fill_between(yc, po, pn, alpha=0.10, color=col)

            sig_cov = "*" if c["p_cov"] < 0.05 else ""
            ax.annotate(
                f"Biais NEMO−OBS : {c['bias_km']:+.0f} km  |  "
                f"σ_N/σ_O : {c['ratio_std']:.2f}  |  "
                f"r_cov : {c['r_cov']:+.2f}{sig_cov}",
                xy=(0.5, 0.96), xycoords="axes fraction",
                ha="center", va="top", fontsize=8,
                color="#444",
                bbox=dict(fc="lightyellow", ec="#aaa", alpha=0.9, pad=2))

    axes[-1].set_xlabel("Année", fontsize=10)

    # Notes en pied de figure
    fig.text(0.99, 0.005, CFG["res_note"],
             ha="right", va="bottom", fontsize=7, color="#a04040",
             style="italic", wrap=True)
    fig.text(0.01, 0.005,
             "Anomalie = φ̃_f(Y) − ⟨φ̃_f⟩  |  "
             "Tendance ✓ = p<0.05 (Student)  |  "
             "r_cov = corrélation des anomalies interann. OBS↔NEMO",
             ha="left", va="bottom", fontsize=7, color="#555", style="italic")

    fname = f"{outdir}regional_{zname}.png"
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")

# =============================================================================
# 7.  FIGURE SYNTHÈSE TOUTES ZONES
# =============================================================================

# Ordre d'affichage des zones (haut → bas) :
#   haut   : Pacifique Sud, Pacifique Est
#   centre : Drake, Campbell, Kerguelen
#   bas    : Fracture Udintsev, Atlantique Sud-Ouest
SUMMARY_ZONE_ORDER = [
    "PacSud", "EPacifique",            # haut
    "Drake", "Campbell", "Kerguelen",  # centre
    "UFZ", "AtlSudOuest",              # bas
]


def fig_summary_all_zones(all_results, outdir):
    """
    Grille compacte : 1 ligne par zone, 1 colonne par front.
    Anomalie en km, Nord vers le haut, OBS trait plein, NEMO tirets.
    L'ordre des zones suit SUMMARY_ZONE_ORDER.
    """
    # Ordre explicite ; on ne garde que les zones réellement analysées,
    # et on ajoute en fin celles éventuellement absentes de la liste.
    zone_keys  = [z for z in SUMMARY_ZONE_ORDER if z in all_results]
    zone_keys += [z for z in all_results if z not in zone_keys]

    n_z = len(zone_keys)
    n_f = len(FRONTS)

    fig, axes = plt.subplots(n_z, n_f, figsize=(5 * n_f, 3.0 * n_z),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for i_z, zk in enumerate(zone_keys):
        zres  = all_results[zk]
        zlab  = zk
        lon1, lon2 = ZONES[zk]["lon"]

        for i_f, front in enumerate(FRONTS):
            ax  = axes[i_z, i_f]
            col = FRONT_COLORS[front]
            plotted = False

            for ds_label, diags_key in [("OBS", "diags_obs"), ("NEMO", "diags_nemo")]:
                diags = zres.get(diags_key, {})
                if front not in diags:
                    continue
                d = diags[front]

                if ds_label == "OBS":
                    anom = d["anom_km"]
                    ls, lw, ms = "-", 1.8, 3.5
                else:
                    ref_obs = zres["diags_obs"].get(front, d)["phi_mean_deg"]
                    anom = (d["phi_deg"] - ref_obs) * DEG2KM
                    ls, lw, ms = "--", 1.2, 2.5

                valid = np.isfinite(anom)
                if valid.sum() < 2:
                    continue
                ax.plot(d["years"][valid], anom[valid],
                        color=col, ls=ls, lw=lw, marker="o", ms=ms,
                        alpha=0.9, label=ds_label)
                plotted = True

            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
            # Nord vers le haut : pas d'inversion d'axe
            ax.grid(True, alpha=0.2)
            ax.axvspan(*CFG["common_period"], color="lightgrey",
                       alpha=0.3, zorder=0)

            if i_z == 0:
                ax.set_title(front, fontsize=11, fontweight="bold", color=col)
            if i_f == 0:
                ax.set_ylabel(f"{zlab}\n({lon1}°–{lon2}°)\nanom. km  (+ = N)",
                              fontsize=11)        # ← agrandi (était 7)
            if i_z == n_z - 1:
                ax.set_xlabel("Année", fontsize=8)
            if plotted:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
            else:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", color="#bbb", fontsize=14)

    plt.tight_layout()
    fname = f"{outdir}regional_summary_all_zones.png"
    plt.savefig(fname, dpi=CFG["fig_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")

# =============================================================================
# 8.  EXPORT CSV
# =============================================================================

def export_csv(all_results, outdir):
    rows = []
    for zk, zres in all_results.items():
        zcfg = ZONES[zk]
        lon1, lon2 = zcfg["lon"]

        for ds_label, diags_key in [("OBS", "diags_obs"), ("NEMO", "diags_nemo")]:
            diags = zres.get(diags_key, {})
            for front, d in diags.items():
                comp = zres.get("comparison", {}).get(front)

                row = {
                    "Zone"            : zk,
                    "Topo"            : zcfg["topo"],
                    "lon_min"         : lon1,
                    "lon_max"         : lon2,
                    "Dataset"         : ds_label,
                    "Front"           : front,
                    "Lat_moy_deg"     : f"{d['phi_mean_deg']:.3f}",
                    "Sigma_km"        : f"{d['sigma_km']:.1f}",
                    "Amplitude_km"    : f"{d['amp_km']:.1f}",
                    "Trend_km_dec"    : f"{d['trend_km_dec']:+.2f}",
                    "Pval_trend"      : f"{d['pval']:.4f}",
                    "Signif_5pct"     : "oui" if d["pval"] < CFG["pval_threshold"] else "non",
                    "N_annees"        : d["n_valid"],
                    "N_lon"           : d["n_lon"],
                }
                if comp and ds_label == "OBS":
                    row["Biais_NEMO_OBS_km"]  = f"{comp['bias_km']:+.1f}"
                    row["Ratio_std_NEMO_OBS"] = f"{comp['ratio_std']:.2f}"
                    row["r_cov"]              = f"{comp['r_cov']:+.3f}"
                    row["p_cov"]              = f"{comp['p_cov']:.4f}"
                    row["N_commun"]           = comp["n_common"]
                else:
                    row["Biais_NEMO_OBS_km"]  = ""
                    row["Ratio_std_NEMO_OBS"] = ""
                    row["r_cov"]              = ""
                    row["p_cov"]              = ""
                    row["N_commun"]           = ""

                rows.append(row)

    df    = pd.DataFrame(rows)
    fname = f"{outdir}regional_summary.csv"
    df.to_csv(fname, index=False, sep=";")
    print(f"  → {fname}")

    # Affichage console
    print(f"\n{'='*72}")
    print("  RÉSUMÉ RÉGIONAL")
    print(f"{'='*72}")
    for zk in all_results:
        zcfg = ZONES[zk]
        print(f"\n  ▶ {zk}  [{zcfg['lon'][0]}°–{zcfg['lon'][1]}°]  {zcfg['topo']}")
        sub = df[df["Zone"] == zk]
        for front in FRONTS:
            sf = sub[sub["Front"] == front]
            if sf.empty:
                continue
            print(f"    {front}")
            for _, row in sf.iterrows():
                sig = "✓" if row["Signif_5pct"] == "oui" else "✗"
                print(f"      {row['Dataset']:<6}  Lat={row['Lat_moy_deg']:>7}°  "
                      f"σ={row['Sigma_km']:>5} km  "
                      f"A={row['Amplitude_km']:>5} km  "
                      f"Trend={row['Trend_km_dec']:>6} km/déc {sig}")
            # Comparaison
            obs_row = sf[sf["Dataset"] == "OBS"]
            if not obs_row.empty and obs_row.iloc[0]["Biais_NEMO_OBS_km"]:
                r = obs_row.iloc[0]
                sig_cov = "✓" if float(r["p_cov"]) < 0.05 else "✗"
                print(f"      OBS↔NEMO  biais={r['Biais_NEMO_OBS_km']:>6} km  "
                      f"σN/σO={r['Ratio_std_NEMO_OBS']}  "
                      f"r_cov={r['r_cov']} {sig_cov}  "
                      f"n={r['N_commun']} ans communs")
    print(f"\n  Note résolution : {CFG['res_note']}")
    print(f"{'='*72}\n")
    return df


# =============================================================================
# 9.  PIPELINE PRINCIPAL
# =============================================================================

def run():
    os.makedirs(CFG["outdir"], exist_ok=True)

    print("=" * 72)
    print("  ANALYSE RÉGIONALE — FRONTS ACC  (V2, sans SAM/ENSO)")
    print(f"  Zones   : {list(ZONES.keys())}")
    print(f"  Sorties : {CFG['outdir']}")
    print("=" * 72)

    # ── Chargement ───────────────────────────────────────────────────────
    print("\n--- Chargement ---")
    ann_obs  = load_annual_positions(CFG["ann_obs_file"],  "OBS")
    ann_nemo = load_annual_positions(CFG["ann_nemo_file"], "NEMO")
    prof_obs  = load_profiles(CFG["profiles_obs"],  "OBS")
    prof_nemo = load_profiles(CFG["profiles_nemo"], "NEMO")

    # ── Aide à la sélection ───────────────────────────────────────────────
    suggest_regions(prof_obs,  "OBS")
    suggest_regions(prof_nemo, "NEMO")

    # ── Figures de sélection (vérification visuelle des bornes) ───────────
    print("\n--- Figures de sélection ---")
    fig_region_selection(prof_obs,  ZONES, "OBS",  CFG["outdir"])
    fig_region_selection(prof_nemo, ZONES, "NEMO", CFG["outdir"])

    # ── Boucle zones ─────────────────────────────────────────────────────
    all_results = {}
    print("\n--- Analyse par zone ---")

    for zname, zcfg in ZONES.items():
        lon1, lon2 = zcfg["lon"]
        print(f"\n  ── {zname}  [{lon1}°, {lon2}°]  |  {zcfg['topo']} ──")

        ts_obs   = build_regional_ts(ann_obs,  lon1, lon2)
        ts_nemo  = build_regional_ts(ann_nemo, lon1, lon2)

        diags_obs  = compute_diagnostics(ts_obs,  f"{zname}/OBS")
        diags_nemo = compute_diagnostics(ts_nemo, f"{zname}/NEMO")

        comparison = compare_obs_nemo(diags_obs, diags_nemo)

        # Console rapide
        for front in FRONTS:
            if front in diags_obs and front in diags_nemo:
                do, dn = diags_obs[front], diags_nemo[front]
                c      = comparison.get(front)
                rat    = f"  σN/σO={c['ratio_std']:.2f}" if c else ""
                print(f"    {front}: OBS σ={do['sigma_km']:.0f} km | "
                      f"NEMO σ={dn['sigma_km']:.0f} km{rat}")

        fig_zone(zname, zcfg, diags_obs, diags_nemo, comparison, CFG["outdir"])

        all_results[zname] = {
            "diags_obs"  : diags_obs,
            "diags_nemo" : diags_nemo,
            "comparison" : comparison,
        }

    # ── Synthèse ─────────────────────────────────────────────────────────
    print("\n--- Figure synthèse ---")
    fig_summary_all_zones(all_results, CFG["outdir"])

    print("\n--- Export CSV ---")
    export_csv(all_results, CFG["outdir"])

    print("\n  Fichiers produits :")
    for f in sorted(os.listdir(CFG["outdir"])):
        print(f"    {CFG['outdir']}{f}")

    return all_results


if __name__ == "__main__":
    run()
