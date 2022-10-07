import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


# set properties first
mpl.rc('text', usetex='True') 
mpl.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "tex"

# load data into the workspace
# active fault shape files were downloaded from 
# 1) https://github.com/cossatot/gem-global-active-faults/tree/master/shapefile
# 2) https://data.humdata.org/dataset/cod-ab-phl
gdf_fault= gpd.read_file('gem_active_faults.shp')
gdf_boundary = gpd.read_file('phl_admbnda_adm3_psa_namria_20200529.shp')
# EQ records acquired from USGS
#
eq = pd.read_csv("bohol_eq_record.csv")

# create geo data frame of EQ records
gdf_eq = gpd.GeoDataFrame(eq, geometry=gpd.points_from_xy(eq.longitude, eq.latitude))

# create maps
fig, ax = plt.subplots(1, 1)

# plot boundaries 
ax2 = gdf_boundary.plot(ax=ax,color='gray')

# plot the active faults
gdf_fault.plot(ax=ax2,color='red',linewidth=1.0,legend=False)
plt.ylim([9.4,10.4])
plt.xlim([123.6,124.7])

# plot EQ
lat, lon = eq['latitude'], eq['longitude']
magn = eq['mag']
plt.plot([127,129],[11,12],c='red',label='Fault')
plt.scatter(lon, lat, label=None,
            c=magn, cmap='viridis',
            s=np.exp(magn)/4, linewidth=0, alpha=0.8)
plt.colorbar(label='Magnitude',orientation='horizontal')
plt.clim(3, 7)

for magn in [4, 5,6,7]:
    plt.scatter([], [], c='k', alpha=0.3, s=np.exp(magn)/4,
                label='M ' +str(magn))

plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", 
           borderaxespad=0,frameon=False,)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquakes in Bohol, Philippines (1974-2021)')
fig.savefig('seismicity_map.png', format='png', dpi=2000, 
            bbox_inches = "tight")
plt.show()
#fig.savefig('seismicity_map.png', format='png', dpi=2000)