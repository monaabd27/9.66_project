import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
p_ap_mean = [0.033821777, 0.255182789, 0.082900495, 0.277950774, 0.162973162, 0.30249444, 0.067307358, 0.243347983, 0.076269175, 0.086840784, 0.242906794, 0.444233919, 0.081700562]
s_ap_mean = [0.12, 0.16, 0.12, 0.16, 0.48, 0.56, 0.16, 0.4, 0.16, 0.52, 0.2, 0.4, 0.24]
# Set position of bar on X axis
barWidth = 0.1
r1 = np.arange(len(p_ap_mean))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
bands =  ["5v1", "5vB", "2v1", "2vB", "1v1", "1vB", "Bv1", "1v2", "Bv2", "Dv5Murd", "5MurdvD", "1v5", "Bv5"]
# Make the plot
plt.bar(r1, p_ap_mean, width=barWidth, edgecolor='white', label='Model')
plt.bar(r2, s_ap_mean, width=barWidth, edgecolor='white', label='Human Data')
# Add xticks on the middle of the group bars
#plt.ylim([0.5, 1])
plt.ylabel('Intention of killing people on side track', fontweight='bold')
plt.xlabel('Scenario', fontweight='bold')
plt.title("Intention Inference (P_harm|A)")
plt.xticks([r + 1/3*barWidth for r in range(len(p_ap_mean))], bands, rotation = 30)
plt.legend()
sns.despine()
# plt.subplot(324)
# plt.scatter(x, y, s=80, c=z, marker=(5, 1))
plt.show()
plt.savefig('intention')