"""
Corrélation transport Drake (NEMO) vs SAM (Marshall, saisonnier) et ENSO (ONI).
Adapté aux fichiers réels :
  - indices_oni_ascii.csv       (ENSO ; sep=';' ; colonnes SEAS YR TOTAL ANOM)
  - newsam_1990_2023_seas.csv   (SAM  ; sep=';' ; colonnes YR ANN AUT WIN SPR SUM)
  - drake_transport_1991_2023.nc

>>> Résolution = SAISONNIÈRE (le SAM n'est pas mensuel). 4 saisons australes/an.
>>> Sans statsmodels (NumPy pur). Lecture locale, aucun téléchargement.

CONVENTION D'ANNÉE (vérifiée sur l'épisode 2015-16) :
  - ONI : DJF[Y] = Déc(Y-1)+Jan(Y)+Fév(Y)            -> étiqueté par l'année de janvier
  - SAM : SUM[Y] = Déc(Y)+Jan(Y+1)+Fév(Y+1)          -> étiqueté par l'année de décembre
  => on décale SUM de +1 an pour que SAM_SUM[Y] s'aligne sur ONI_DJF[Y+1].
  MAM/JJA/SON : pas d'ambiguïté (une seule année civile).

Sorties :
  - drake_index_correlation.csv
  - drake_seasonal_overlay.png
  - drake_seasonal_stratified.png
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, t as student_t

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # dossier du script

FILE_TRANSPORT = os.path.join(HERE, "drake_transport_1991_2023.nc")
VAR_TRANSPORT  = "drake_transport"
FILE_ONI       = os.path.join(HERE, "indices_oni.ascii.csv")
FILE_SAM       = os.path.join(HERE, "newsam_1990_2023_seas.csv")

SEASONS    = ["DJF", "MAM", "JJA", "SON"]
SEAS_MONTH = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}   # mois repère pour l'axe temps
HAC_LAGS   = 4                                            # ~1 an (saisons)


# ─────────────────────────────────────────────────────────────────
# LECTURE DES FICHIERS
# ─────────────────────────────────────────────────────────────────
def load_transport_monthly():
    ds = xr.open_dataset(FILE_TRANSPORT)
    s = ds[VAR_TRANSPORT].to_series()
    s.index = pd.DatetimeIndex(ds["time_counter"].values).to_period("M").to_timestamp()
    return s.rename("T")

def transport_to_seasonal(s):
    df = s.to_frame()
    mo, yr = df.index.month, df.index.year
    seas = np.select(
        [np.isin(mo, [1, 2]), mo == 12, np.isin(mo, [3, 4, 5]),
         np.isin(mo, [6, 7, 8]), np.isin(mo, [9, 10, 11])],
        ["DJF", "DJF", "MAM", "JJA", "SON"], default="")
    yc = np.where(mo == 12, yr + 1, yr)            # déc -> saison DJF de l'année suivante
    df["season"], df["year"] = seas, yc
    g = (df.groupby(["year", "season"])["T"]
           .agg(["mean", "count"]).reset_index())
    g = g[g["count"] >= 2].rename(columns={"mean": "T"})    # >=2 mois valides
    return g[["year", "season", "T"]]

def load_oni_seasonal():
    d = pd.read_csv(FILE_ONI, sep=";", encoding="utf-8-sig")
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d = d[d["SEAS"].isin(SEASONS)].rename(
        columns={"YR": "year", "SEAS": "season", "ANOM": "ENSO"})
    return d[["year", "season", "ENSO"]]

def load_oni_annual():
    d = pd.read_csv(FILE_ONI, sep=";", encoding="utf-8-sig")
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    return d.groupby("YR")["ANOM"].mean().rename("ENSO")

def load_sam_seasonal():
    d = pd.read_csv(FILE_SAM, sep=";", encoding="utf-8-sig")
    m = {"AUT": "MAM", "WIN": "JJA", "SPR": "SON", "SUM": "DJF"}
    long = d.melt(id_vars="YR", value_vars=list(m),
                  var_name="aus", value_name="SAM")
    long["season"] = long["aus"].map(m)
    long["year"]   = long["YR"] + (long["season"] == "DJF").astype(int)  # +1 sur DJF
    return long[["year", "season", "SAM"]]

def load_sam_annual():
    d = pd.read_csv(FILE_SAM, sep=";", encoding="utf-8-sig")
    return d.set_index("YR")["ANN"].rename("SAM")


# ─────────────────────────────────────────────────────────────────
# STATS (NumPy / SciPy)
# ─────────────────────────────────────────────────────────────────
def effective_n(x, y):
    n = len(x)
    rx, ry = pd.Series(x).autocorr(1), pd.Series(y).autocorr(1)
    if np.isnan(rx) or np.isnan(ry):
        return float(n)
    return float(max(3.0, min(n * (1 - rx * ry) / (1 + rx * ry), n)))

def corr_with_pcorr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p_raw = pearsonr(x, y)
    neff = effective_n(x, y)
    tval = r * np.sqrt((neff - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    p_eff = 2 * (1 - student_t.cdf(abs(tval), df=neff - 2))
    return r, p_raw, p_eff, neff

def ols_newey_west(y, X, maxlags):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    h = X * resid[:, None]
    S = h.T @ h
    for l in range(1, maxlags + 1):
        w = 1.0 - l / (maxlags + 1)
        G = h[l:].T @ h[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * (1 - student_t.cdf(np.abs(tval), df=n - k))
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    r2adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return beta, se, pval, r2, r2adj


# ─────────────────────────────────────────────────────────────────
# PROGRAMME
# ─────────────────────────────────────────────────────────────────
def main():
    # --- construction du tableau saisonnier ---------------------------
    tr   = transport_to_seasonal(load_transport_monthly())
    sam  = load_sam_seasonal()
    oni  = load_oni_seasonal()
    df = (tr.merge(sam, on=["year", "season"])
            .merge(oni, on=["year", "season"]))
    df["date"] = pd.to_datetime(dict(year=df.year,
                                     month=df.season.map(SEAS_MONTH), day=1))
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Points saisonniers fusionnés : {len(df)} "
          f"({df.year.min()}–{df.year.max()})")

    # anomalie de transport = on retire la moyenne propre à chaque saison
    df["Ta"] = df["T"] - df.groupby("season")["T"].transform("mean")

    rows = []
    # --- A. corrélations saisonnières globales (toutes saisons) -------
    for idx in ["SAM", "ENSO"]:
        r, pr, pe, ne = corr_with_pcorr(df[idx], df["Ta"])
        rows.append(["global", idx, r, r**2, pr, pe, ne, len(df)])

    # --- B. corrélations stratifiées par saison ----------------------
    strat = {}
    for s in SEASONS:
        sub = df[df.season == s]
        for idx in ["SAM", "ENSO"]:
            r, pr, pe, ne = corr_with_pcorr(sub[idx], sub["T"])
            rows.append([s, idx, r, r**2, pr, pe, ne, len(sub)])
            strat[(s, idx)] = r

    # --- C. régression multiple Ta ~ SAM + ENSO (Newey-West) ---------
    y = df["Ta"].values
    X = np.column_stack([np.ones(len(df)), df["SAM"].values, df["ENSO"].values])
    beta, se, pval, r2, r2adj = ols_newey_west(y, X, HAC_LAGS)
    print("\n── Régression saisonnière  Ta = c + a·SAM + b·ENSO  (Newey-West) ──")
    for nm, b, sdv, p in zip(["const", "SAM", "ENSO"], beta, se, pval):
        print(f"  {nm:5s}: {b:+.3f}  (se={sdv:.3f}, p={p:.2g})")
    print(f"  R² = {r2:.3f}   R² ajusté = {r2adj:.3f}")

    # --- D. robustesse annuelle --------------------------------------
    tr_ann  = load_transport_monthly().groupby(lambda d: d.year).mean().rename("T")
    ann = pd.concat({"T": tr_ann, "SAM": load_sam_annual(),
                     "ENSO": load_oni_annual()}, axis=1).dropna()
    ann["Ta"] = ann["T"] - ann["T"].mean()
    print(f"\n── Robustesse annuelle ({ann.index.min()}–{ann.index.max()}, "
          f"n={len(ann)}) ──")
    for idx in ["SAM", "ENSO"]:
        r, pr, pe, ne = corr_with_pcorr(ann[idx], ann["Ta"])
        rows.append(["annuel", idx, r, r**2, pr, pe, ne, len(ann)])
        print(f"  {idx}: r={r:+.2f}  R²={r**2:.2f}  p_corr={pe:.2g}")
    Xa = np.column_stack([np.ones(len(ann)), ann["SAM"], ann["ENSO"]])
    ba, _, pa, r2a, _ = ols_newey_west(ann["Ta"].values, Xa, 1)
    print(f"  régression annuelle : R²={r2a:.2f} "
          f"(SAM {ba[1]:+.2f} p={pa[1]:.2g} | ENSO {ba[2]:+.2f} p={pa[2]:.2g})")

    # --- sauvegarde table --------------------------------------------
    res = pd.DataFrame(rows, columns=["echelle", "indice", "r", "R2",
                                      "p_brut", "p_corrige", "N_eff", "n"])
    res.to_csv("drake_index_correlation.csv", index=False)
    print("\n── Table des corrélations ───────────────────────────────")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # --- FIG 1 : séries saisonnières superposées ---------------------
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(df.date, df["Ta"], color="steelblue", lw=1.6, label="T' saisonnier (Sv)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("Anomalie de transport T' (Sv)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(df.date, df["SAM"],  color="tab:green", lw=1.1, alpha=.8, label="SAM")
    ax2.plot(df.date, df["ENSO"], color="tab:red",   lw=1.1, alpha=.8, label="ENSO (ONI)")
    ax2.set_ylabel("Indices SAM / ENSO", color="dimgray")
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax1.set_title("Transport Drake (saisonnier) vs SAM et ENSO")
    fig.tight_layout(); fig.savefig("drake_seasonal_overlay.png", dpi=150,
                                    bbox_inches="tight")

    # --- FIG 2 : corrélations stratifiées par saison -----------------
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SEASONS)); w = 0.38
    ax.bar(x - w/2, [strat[(s, "SAM")]  for s in SEASONS], w,
           label="SAM",  color="tab:green")
    ax.bar(x + w/2, [strat[(s, "ENSO")] for s in SEASONS], w,
           label="ENSO", color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel("Corrélation r(indice, transport)")
    ax.set_title("Corrélation transport–indice par saison australe")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout(); fig2.savefig("drake_seasonal_stratified.png", dpi=150,
                                      bbox_inches="tight")
    print("\n✓ CSV et figures sauvegardés."); plt.show()


if __name__ == "__main__":
    main()