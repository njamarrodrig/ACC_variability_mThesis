import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpathmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath


ds = xr.open_dataset('zos_1991_2023_IS.nc')
zos_mean = ds['zos'].mean(dim='time_counter')
zos_mean = zos_mean.where(zos_mean != 0)

lat = ds['nav_lat']
lon = ds['nav_lon']

def remap(lon, lat, field):
    nx, ny = lon.shape
    new_lon = np.zeros([nx, ny])
    new_lat = np.zeros([nx, ny])
    new_field = np.zeros([nx, ny])
    for i in range(nx):
        j = np.argmin(lon[i, :])
        new_lon[i, :]   = np.append(lon[i, j:],   lon[i, :j])
        new_lat[i, :]   = np.append(lat[i, j:],   lat[i, :j])
        new_field[i, :] = np.append(field[i, j:], field[i, :j])
    return new_lon, new_lat, new_field

lon1, lat1, zos1 = remap(lon.values, lat.values, zos_mean.values)

fig = plt.figure(figsize=(10, 10))
proj = ccrs.SouthPolarStereo()
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

# ── Masque circulaire ──────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
# ──────────────────────────────────────────────────────────────────

im = ax.pcolormesh(
    lon1, lat1, zos1,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    shading='nearest',
    vmin=-2.5, vmax=0
)

cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.05)
cbar.set_label('SSH (m)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

levels = np.arange(-2.5, 0.1, 0.1)
ax.contour(
    lon1, lat1, zos1,
    levels=levels,
    colors='k',
    linewidths=0.5,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', color='black', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.gridlines(linewidth=0.4, linestyle='--', color='grey')

t_min = str(ds['time_counter'].values[0])[:7]
t_max = str(ds['time_counter'].values[-1])[:7]
ax.set_title(
    f'Sea Surface Height – Moyenne {t_min} / {t_max}\n',
    fontsize=13, fontweight='bold', pad=20
)

plt.savefig("ZOS_mean.png", dpi=300, bbox_inches='tight')
plt.show()