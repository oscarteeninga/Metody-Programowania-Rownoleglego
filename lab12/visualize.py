import numpy as np
import matplotlib.pyplot as plt
import os
import math

files = os.listdir('./temperature')

for file in files:
    with open("./temperature/" + file) as f:
        data = f.read()
        u_str = [row.split("\t")[:-1] for row in data.split('\n')[:-1]]
        # Skala logarytmiczna, zeby bylo ladniej widac
        u_f = [[math.log(float(x)+1) for x in xs] for xs in u_str]
        plt.imsave("./visualize/" + file.replace(".txt", ".png"), u_f)
