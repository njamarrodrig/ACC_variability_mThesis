
## Structure du dépôt pour le mémoire : Variabilité des fronts du Courant Circumpolaire Antarctique (ACC)
 
### Position moyenne
| Script | Rôle |
|--------|------|
| `MEAN_SSH_OBS.py` | Carte de la DOT moyenne (observations) en projection polaire stéréographique |
| `MEAN_SSH_NEMO.py` | Carte de la SSH moyenne (modèle NEMO), avec ré-arrangement longitudinal de la grille |
 
### Détection et variabilité des fronts
 
| Script | Rôle |
|--------|------|
| `FRONT_VARIABILITY_V5.py` | Pipeline principal. sétection des fronts sur moyennes annuelles de SSH, correction d'un mode commun large échelle, séries temporelles, cartes de variabilité, exports CSV/NetCDF |
| `FRONT_VAR_MEAN_PLUS.py` | Comparaison méthodologique : niveaux SSH fixes vs annuels recalibrés, validation par gradient SSH et vitesse géostrophique, désaisonnalisation |
| `FRONT_REGIONAL_V1.py` | Analyse de la variabilité régionale par zones géographiques (bassin Pacifique Sud, zone de fracture d'Udintsev, dorsale Est-Pacifique…) |
| `FRONT_SCHIAVON_NEMO.py` | Identification des fronts selon Sokolov & Rintoul (2009a) : histogrammes SSH pondérés par le gradient `|∇η|`, par secteurs longitudinaux |
 
### Validation et modes climatiques
 
| Script | Rôle |
|--------|------|
| `VALIDATION_FRONT.py` | Comparaison des fronts calculés (OBS + modèle) avec les fronts de la littérature |
| `VALIDATION_FRONT_SR2009.py` | Validation des fronts S&R 2009a sur NEMO : cartes du gradient SSH moyen et superposition à la littérature |
| `FRONT_circum_ENSO_SAM.py` | Séries temporelles circumpolaires et profils longitudinaux croisés avec les phases ENSO et SAM |
 

 
Scripts en Python 3 :
```bash
pip install numpy xarray pandas scipy matplotlib cartopy shapely netCDF4
```
 
mémoire de N. Jamar Rodriguez
