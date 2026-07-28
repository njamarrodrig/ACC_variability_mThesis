
## Structure du dépôt pour le mémoire : Variabilité des fronts du Courant Circumpolaire Antarctique (ACC)
 
| Fichier | Description |
|---|---|
| `JamarRodriguez_32251900_2026.pdf` | Document mémoire complet. |
| `MEAN_SSH_NEMO.py` | Carte de la SSH moyenne (1991–2023) issue du modèle NEMO, projection polaire sud. |
| `MEAN_SSH_OBS.py` | Carte de la SSH moyenne (2002–2018) des observations satéllitaire, projection polaire sud. |
| `FRONT_SCHIAVON_NEMO.py` | Identification des fronts (SAF, PF, SACCF) dans NEMO par la méthode Sokolov & Rintoul (2009a)|
| `FRONT_PARK_OBSNEMO.py` | Identification des fronts par la méthode Park et al. (2019) pour OBS et NEMO. |
| `FRONT_VARIABILITY_V5.py` | Calcul de la variabilité interannuelle des fronts : positions annuelles, séries temporelles circumpolaires, correction du mode large-échelle SSH. |
| `FRONT_REGIONAL_V1.py` | Variabilité régionale des fronts : profils longitudinaux σ(λ), biais OBS/NEMO, séries temporelles zonales. |
| `FRONT_circum_ENSO_SAM.py` | Séries temporelles des positions circumpolaires annotées par phase ENSO et SAM ; profils longitudinaux composites par phase pour évaluer l'influence régionale vs circumpolaire. |
| `VALIDATION_FRONT.py` | Comparaison des fronts calculés avec des trajectoires de la littérature (GeoJSON). |
| `VALIDATION_FRONT_SR2009.py` | Comparaison des deux méthodes (S&R 2009a vs Park 2019) appliquées à NEMO. |
| `PROXY_NBSB_TRANSP_VAR.py` | Calcul du proxy de transport : gradient SSH entre NB et SB de l'ACC au passage de Drake, pour OBS et NEMO. |
| `NEMO_PROXY_VS_TRANSP.py` | Corrélation entre le transport à Drake (NEMO) et le gradient SSH NB−SB : séries temporelles, nuage de points, calibration en Sv/m. |
| `TRANSPORT_SVD_NEMO.py` | Corrélation du transport à Drake avec les indices SAM et ENSO (saisonnier et annuel), régression multiple avec correction |


Scripts en Python 3 :
```bash
pip install numpy xarray pandas scipy matplotlib cartopy shapely netCDF4
```
 
mémoire de N. Jamar Rodriguez
