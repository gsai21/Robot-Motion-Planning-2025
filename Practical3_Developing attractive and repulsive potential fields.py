import numpy as np
import matplotlib.pyplot as plt

# Grid tayaruchesukovadam
x_m, y_m = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
lakshyam = np.array([9, 9])
aapagaatham = np.array([5, 5])

# Aakarshana potential – parabola akaralo
U_akarshana = 0.5 * ((x_m - lakshyam[0])**2 + (y_m - lakshyam[1])**2)

# Repulsion potential – prabhava vyasthamu ki lopala matrame
eta_gatham = 100     # scale chese factor
d_bahudooram = 1.5   # prabhava vyasthamu
dooram_obs = np.sqrt((x_m - aapagaatham[0])**2 + (y_m - aapagaatham[1])**2)
U_vikarshana = np.where(
    dooram_obs <= d_bahudooram,
    0.5 * eta_gatham * ((1/dooram_obs - 1/d_bahudooram)**2),
    0
)

# Mottham potential kalipi
U_mottham = U_akarshana + U_vikarshana

# Negative gradient ganakam – disha sūcana kosam
grad_y, grad_x = np.gradient(-U_mottham)

# Chitram tayaruchesukovadam
fig, aks = plt.subplots(figsize=(8, 8))
rekhalu = aks.contourf(x_m, y_m, U_mottham, levels=100, cmap='plasma')
bar = plt.colorbar(rekhalu, ax=aks, label='Potential')
aks.contour(x_m, y_m, U_mottham, levels=30, colors='black', linewidths=0.3, alpha=0.3)

# Quiver – robot yoka gamana disha chupinchadam
vichchedham = 5
aks.quiver(
    x_m[::vichchedham, ::vichchedham],
    y_m[::vichchedham, ::vichchedham],
    grad_x[::vichchedham, ::vichchedham],
    grad_y[::vichchedham, ::vichchedham],
    color='white', alpha=0.6, width=0.002
)

# Lakshyam mariyu aapagaatham sthalalu
aks.plot(lakshyam[0], lakshyam[1], 'ro', markersize=10, label='Goal')
aks.plot(aapagaatham[0], aapagaatham[1], 'kx', markersize=10, label='Obstacle')

# Aakarshanalu mariyu alankarana
aks.set_title("Improved Potential Field Visualization", fontsize=14)
aks.set_xlabel("X")
aks.set_ylabel("Y")
aks.set_aspect('equal')
aks.grid(True, linestyle='--', alpha=0.4)
aks.legend(loc='upper left')
plt.tight_layout()
plt.show()
