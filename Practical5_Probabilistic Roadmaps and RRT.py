import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def dhooram(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def yadhruicha_bindhu():
    return np.random.uniform(0, 10), np.random.uniform(0, 10)

prarambham = (1, 1)
lakshyam = (9, 9)
mokalu = [prarambham]
rekhulu = []
sadhyam_ayindi = False

# RRT node expansion loop
for _ in range(500):
    chinna = yadhruicha_bindhu()
    daggara = min(mokalu, key=lambda n: dhooram(n, chinna))
    disha = np.array(chinna) - np.array(daggara)
    chinna_adi = 0.5
    disha = disha / np.linalg.norm(disha)
    kotta = tuple(np.array(daggara) + chinna_adi * disha)

    # Skip if outside boundary
    if not (0 <= kotta[0] <= 10 and 0 <= kotta[1] <= 10):
        continue

    mokalu.append(kotta)
    rekhulu.append((daggara, kotta))

    if dhooram(kotta, lakshyam) < 0.7:
        rekhulu.append((kotta, lakshyam))
        sadhyam_ayindi = True
        break

# Visualization
fig, aksharam = plt.subplots(figsize=(7, 7))
for (m1, m2) in rekhulu:
    aksharam.plot([m1[0], m2[0]], [m1[1], m2[1]], 'cornflowerblue', linewidth=0.8)

# Highlight the final edge to goal
if sadhyam_ayindi:
    aksharam.plot([kotta[0], lakshyam[0]], [kotta[1], lakshyam[1]],
                  'orange', linewidth=2, label='Path to Goal')

# Markers for start and goal
aksharam.plot(prarambham[0], prarambham[1], 'go', markersize=10, label='Start')
aksharam.plot(lakshyam[0], lakshyam[1], 'ro', markersize=10, label='Goal')

# Formatting
aksharam.set_title("Rapidly-Exploring Random Tree (RRT)", fontsize=14)
aksharam.set_xlim(0, 10)
aksharam.set_ylim(0, 10)
aksharam.set_aspect('equal')
aksharam.grid(True, linestyle='--', linewidth=0.5)
aksharam.legend(loc='upper left')
plt.tight_layout()
plt.show()