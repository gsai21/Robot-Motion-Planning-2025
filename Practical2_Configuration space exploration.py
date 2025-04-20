import matplotlib.pyplot as plt

# Asalaina aapaghatamulu (x1, y1) = kinda-edama, (x2, y2) = paina-kudama
aadya_addankalu = [((2, 2), (3, 4)), ((6, 5), (7, 7))]
robotu_vyaasaartham = 0.5

# Addankalu vyastharanga perigina roopam lo marchadam
def vistrutam_chesina_addankalu(addankalu, vyaasam):
    vistrutam = []
    for (x1, y1), (x2, y2) in addankalu:
        vistrutam.append(((x1 - vyaasam, y1 - vyaasam), (x2 + vyaasam, y2 + vyaasam)))
    return vistrutam

# Chitra setup
chitra, chakram = plt.subplots(figsize=(8, 8))
chakram.set_title("Configuration Space with Inflated Obstacles")
chakram.set_xlabel("X")
chakram.set_ylabel("Y")
chakram.set_aspect('equal')
chakram.grid(True)

# Aadya addankalu chitram
for (x1, y1), (x2, y2) in aadya_addankalu:
    rekta = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', label='Original Obstacle')
    chakram.add_patch(rekta)

# Vistrutam aapaghatamulu chitram (C-space)
vistrutam_addankalu = vistrutam_chesina_addankalu(aadya_addankalu, robotu_vyaasaartham)
for sankhya, ((x1, y1), (x2, y2)) in enumerate(vistrutam_addankalu):
    rekta = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='red', alpha=0.3,
                          label='Inflated Obstacle' if sankhya == 0 else "")
    chakram.add_patch(rekta)

# Icha anusaranga: prarambha sthalamlo robotu chitram
robotu_sthalam = (1, 1)
robotu_gola = plt.Circle(robotu_sthalam, robotu_vyaasaartham, color='blue', alpha=0.5, label='Robot')
chakram.add_patch(robotu_gola)
chakram.plot(*robotu_sthalam, 'bo')  # robotu kendram

# Legend tayaru cheyyadam
rekhalu, sanketalu = chakram.get_legend_handles_labels()
thirigi_lekka = dict(zip(sanketalu, rekhalu))  # punarukti sanketalu teeseseyadam
chakram.legend(thirigi_lekka.values(), thirigi_lekka.keys(), loc='upper right')

plt.show()
