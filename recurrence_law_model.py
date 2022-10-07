import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy as sci
from scipy import optimize

plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('font', size=14)          # controls default text sizes
plt.rcParams["font.family"] = "tex"
mpl.rc('text', usetex='True') 

# EQ records acquired from USGS(https://earthquake.usgs.gov/earthquakes/search/)
eq = pd.read_csv("bohol_eq_record.csv")

# add year column in dataframe
time = eq['time'][:]
year = []
for i in range(0,len(time)):
    year.append(float(time[i][0:4]))
eq['year'] = year

# calculate annual exceedance
magni = eq['mag']
m = np.arange(4, 7.5, 0.5)
time_period = eq['year'].max() - eq['year'].min() 
count_gt_mag,lambda_m = [],[]
for lim in m:
    count_m = magni[magni >= lim].count()
    count_gt_mag.append(count_m)
    lambda_m.append(count_m /time_period )

# plot annual exceedance
fig, ax = plt.subplots(1,1)
plt.plot(m,lambda_m,'k--^')
plt.xlabel('Magnitude')
plt.ylabel('$\lambda_{\geq m_i}$')
plt.title('Annual rate of EQ exceedance')
plt.show()
fig.savefig('annual_exceed.png', 
            format='png', dpi=2000, bbox_inches = "tight")

# fit a linear model using G-R law
yi = np.log(lambda_m)
xi = m
model = np.polyfit(xi,yi,1)
b = -model[0]
a = model[1]

mi = np.linspace(4,7,10)
lambda_mi = model[0]*mi + model[1]

# Plot recorded data and G-R model
fig2, ax2 = plt.subplots(1,1)
fig2.set_size_inches(4,4)
plt.plot(m,10**np.log(lambda_m),'o',label='Data')
plt.plot(mi,10**lambda_mi,label=f'G-R: $\log \lambda_m$ = {a:.3f}$-${b:.3f}m')
plt.xlim([4,7])
ax2.set_yscale('log')
ax2.legend(shadow='True')
plt.xlabel('Magnitude')
plt.ylabel('$\lambda_m$')
plt.grid(linestyle='-.')
plt.title('EQ Recurrence Relationship for Bohol, Phl')
plt.show()
fig2.savefig('GR_model.png', format='png', dpi=2000, bbox_inches = "tight")

# Define function of Bounded G-R
def func(x,a,b):
    m0,mmax = 4.0,7.1
    m_range = mmax-m0
    alpha,beta = 2.303*a,2.303*b 
    nu = np.exp(alpha - beta*m0)
    lambda_m = nu * (np.exp(-beta*(x-m0))-np.exp(-beta*m_range))/(1-np.exp(-beta*m_range))
    return lambda_m

popt, pcov = sci.optimize.curve_fit(func, m, lambda_m)

## Plot recorded data and Bounded G-R model
fig3,ax3 = plt.subplots(1,1)
fig3.set_size_inches(4,4)
plt.plot(m, 10**np.log(lambda_m), 'o', label='Data')
plt.plot(m, 10**np.log(func(m, *popt)),
         label=f'Bounded G-R: a = {popt[0]:.3f}, b = {popt[1]:.3f}')
ax3.set_yscale('log')
ax3.legend(shadow='True')
plt.xlabel('Magnitude')
plt.ylabel('$\lambda_m$')
plt.grid(linestyle='-.')
plt.xlim([4,7])
plt.title('EQ Recurrence Relationship for Bohol, Phl')
plt.show()
fig.savefig('BoundedGR_model.png', format='png', dpi=2000, bbox_inches = "tight")

# determine annual rate for each range of EQ magnitude
magnitude_range = np.zeros((4,1))
for i in range(0, len(eq['mag'] )):
    if eq['mag'][i] >= 4.0 and eq['mag'][i] < 5.0:
        magnitude_range[0] = magnitude_range[0] + 1
    if eq['mag'][i] >= 5.0 and eq['mag'][i] < 6.0:
        magnitude_range[1] = magnitude_range[1] + 1
    if eq['mag'][i] >= 6.0 and eq['mag'][i] < 7.0:
        magnitude_range[2] = magnitude_range[2] + 1
    if eq['mag'][i] >= 7.0 and eq['mag'][i] < 8.0:
        magnitude_range[3] = magnitude_range[3] + 1
annual_rate = magnitude_range / (eq['year'].max() - eq['year'].min()) 
mag_range = [['[4.0,5.0)',float(annual_rate[0])],
             ['[5.0,6.0)',float(annual_rate[1])],
             ['[6.0,7.0)',float(annual_rate[2])],
             ['[7.0,8)',float(annual_rate[3])]]
ann_rate = pd.DataFrame(mag_range,columns=['Magnitude Range','Annual Rate'])

# Plot annual rate of EQ magnitude
myplot=ann_rate.plot(x='Magnitude Range', y='Annual Rate',
                     c='k',linestyle='--',marker='X',
                     legend=False,title='Annual Rate of EQ Magnitude')
myplot.set_ylabel('$\lambda_{m=m_i}$')
fig4=myplot.get_figure()
fig4.savefig('annual_rate.png', format='png', dpi=2000, bbox_inches = "tight")
