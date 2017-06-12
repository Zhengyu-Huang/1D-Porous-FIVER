__author__ = 'zhengyuh'
import numpy as np
import matplotlib.pyplot as plt
V_Rie = np.load('Propeller_Riemann.npy')
V_old = np.load('Propeller_old_source.npy')
V_new = np.load('Propeller_new_source.npy')
L =4.0
N = 500
x = np.linspace(0, L, N)
plt.figure(1)
plt.plot(x, V_new[:, 1], 'ro-',markersize=2.0, label = 'Source term')
plt.plot(x, V_Rie[:, 1], 'o-',markersize=2.0, label = 'Riemann solver')
#plt.plot(x, V_old[:, 1], 'o-',markersize=2.0, label = 'Old source term')
plt.tick_params(labelsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('Velocity(m/s)',fontsize=15)
plt.title('Velocity',fontsize=20)
plt.legend(prop={'size':15})
plt.tight_layout()
plt.grid(linestyle='dashed')


plt.figure(2)
plt.plot(x, V_new[:, 2], 'ro-',markersize=2.0, label = 'Source term')
plt.plot(x, V_Rie[:, 2], 'o-',markersize=2.0, label = 'Riemann solver')
#plt.plot(x, V_old[:, 2], 'o-',markersize=2.0, label = 'Old source term')
plt.tick_params(labelsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('Pressure(Pa)',fontsize=15)
plt.title('Pressure',fontsize=20)
plt.legend(prop={'size':15})
plt.tight_layout()
plt.grid(linestyle='dashed')
plt.show()


