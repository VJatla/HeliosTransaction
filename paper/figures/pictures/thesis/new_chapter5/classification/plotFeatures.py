import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt




font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('text', usetex=True)

objects = ('newN', 'newA', 'missN', 'missA', 'overA','sameA')
y_pos = np.arange(len(objects))
performance = [0.02,0.03,0.06,0.3,0.19,0.10]
barWidth = 0.5
plt.figure(figsize=(6,4.5), dpi=1000)
plt.bar(y_pos, performance, barWidth, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.xlabel(r'\texbf{Features}', fontsize=18)
plt.ylabel(r'\texbf{Features relative importance}', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('classificationFeatures.png')
