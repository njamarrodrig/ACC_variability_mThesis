# %% import libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %% load data
ds = xr.open_dataset('dot_all_30bmedian_eigen6s4v2_sig3.nc')

# %% moyenne temporelle
dot_mean = ds['dot'].mean(dim='time')
dot_mean = dot_mean.where(ds['land_mask'] == 0)
dot_mean = dot_mean.transpose('latitude', 'longitude')  # → (lat, lon)

lat = ds['latitude'].values
lon = ds['longitude'].values
LON, LAT = np.meshgrid(lon, lat)

# %% plot
fig = plt.figure(figsize=(10, 10))
proj = ccrs.SouthPolarStereo()
ax = plt.axes(projection=proj)
ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

# ── Masque circulaire ──────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 100)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)
# ──────────────────────────────────────────────────────────────────

im = ax.pcolormesh(
    LON, LAT, dot_mean.values,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    shading='nearest',
    vmin=-2.5, vmax=0
)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.05)
cbar.set_label('DOT (m)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Contours
levels = np.arange(-2.5, 0.1, 0.1)
ax.contour(
    LON, LAT, dot_mean.values,
    levels=levels,
    colors='k',
    linewidths=0.8,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', color='black', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.gridlines(linewidth=0.4, linestyle='--', color='grey')

# Titre dynamique
t_min = str(ds['time'].values[0])[:7]
t_max = str(ds['time'].values[-1])[:7]
ax.set_title(
    f'Dynamic Ocean Topography – Moyenne {t_min} / {t_max}\n(Océan Austral, < 50°S)',
    fontsize=13, fontweight='bold', pad=20
)

plt.savefig("DOT_mean.png", dpi=300, bbox_inches='tight')
plt.show()