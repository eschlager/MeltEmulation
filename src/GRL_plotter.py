# -*- coding: utf-8 -*-
"""
Created on 01.12.2025
@author: eschlager
plotter function to plot data over Greenland with proper projection and coastline
"""


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.path import Path
from cartopy.feature import ShapelyFeature
from shapely.geometry import shape, mapping


def find_lon_lat(ds):
    # Try common names (coords or data_vars)
    lon_names = [n for n in ds.coords if n.lower() in ('lon', 'longitude', 'long')]
    lat_names = [n for n in ds.coords if n.lower() in ('lat', 'latitude')]
    if not lon_names or not lat_names:
        raise ValueError("Couldn't find lon/lat variables!")
    else:
        lon = ds[lon_names[0]].values
        lat = ds[lat_names[0]].values
        return lon, lat


def normalize_lons(lon):
    """Convert 0..360 to -180..180 if needed."""
    lon = np.array(lon)
    if lon.min() >= 0 and lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def load_greenland_shape(resolution='10m'):
    """
    Load the Natural Earth 'admin_0_countries' and return the shapely geometry
    for Greenland (Polygon or MultiPolygon).
    """
    shpname = shapereader.natural_earth(resolution=resolution, category='cultural', name='admin_0_countries')
    reader = shapereader.Reader(shpname)
    for rec in reader.records():
        props = rec.attributes
        name = props.get('NAME') or props.get('NAME_LONG') or props.get('ADMIN')
        if name and name.lower() == 'greenland':
            geom = rec.geometry  # shapely geometry
            return geom
    raise RuntimeError("Greenland geometry not found in Natural Earth data.")


def points_in_polygons(lon, lat, polygon):
    """
    lon, lat : arrays (2D) of same shape
    polygon : shapely geometry (Polygon or MultiPolygon)
    Returns boolean mask (2D) True where point is inside the polygon.
    Uses matplotlib.path.Path for each polygon part for speed.
    """
    pts = np.column_stack((lon.ravel(), lat.ravel()))
    inside = np.zeros(pts.shape[0], dtype=bool)
    # polygon may be MultiPolygon or Polygon
    if polygon.geom_type == 'Polygon':
        polys = [polygon.exterior.coords]
        # include interiors (holes) as negative masks
        interiors = list(polygon.interiors)
    else:
        polys = [poly.exterior.coords for poly in polygon.geoms]
        interiors = [interior for poly in polygon.geoms for interior in poly.interiors]

    for poly_coords in polys:
        path = Path(np.array(poly_coords))
        inside |= path.contains_points(pts)

    # subtract holes
    for interior in interiors:
        path_h = Path(np.array(interior.coords))
        inside &= ~path_h.contains_points(pts)

    return inside.reshape(lon.shape)


def plot_greenland_only(gris_data, ax=None, masked=True, ocean_color='white', gridlines=False, pcolormesh_kwargs=None):
    ds = gris_data

    if ds.ndim != 2:
        raise ValueError("Expected a 2D field (lat,lon or y,x).")

    lon, lat = find_lon_lat(ds)

    # If lon/lat are 1D, make meshgrid
    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    elif lon.shape == ds.shape and lat.shape == ds.shape:
        Lon, Lat = lon, lat
    else:
        raise ValueError(f"lon/lat shapes {lon.shape},{lat.shape} do not match data shape {ds.shape}")

    # Normalize longitudes to -180..180 to match Natural Earth polygons
    Lon = normalize_lons(Lon)
    # load Greenland shapely geometry
    greenland_geom = load_greenland_shape(resolution='50m')

    # Create mask of points inside Greenland polygons (using cell centers)
    inside_mask = points_in_polygons(Lon, Lat, greenland_geom)

    # Mask the data outside Greenland
    data = ds.values
    if masked:
        data = np.ma.array(data, mask=~inside_mask)

    # Plot
    display_crs = ccrs.NorthPolarStereo(central_longitude=-40)
    data_crs = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=(8,10), frameon=False)
        ax = fig.add_subplot(1,1,1, projection=display_crs)
        plt.axis('off')

    # Make ocean white if not None, otherwise default is transparent!
    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=ocean_color, zorder=0)
    # Fill Greenland with light gray (use ShapelyFeature for correct reprojection)
    greenland_feature = ShapelyFeature([greenland_geom], crs=ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black', linewidth=.5, zorder=3)
    ax.add_feature(greenland_feature)

    # Set a Greenland-focused extent in lon/lat
    ax.set_extent((-55, -25, 58, 85), crs=ccrs.PlateCarree())

    # Latitude gridlines visible
    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.7, linestyle='--')

    pcm = ax.pcolormesh(Lon, Lat, data, transform=data_crs, shading='auto', rasterized=True, zorder=4, **pcolormesh_kwargs)

    return ax, pcm