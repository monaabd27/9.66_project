import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
p_ap_mean = [0.738835066386733, 0.5085295445985663, 0.7415065282234399, 0.5159192780837973, 0.678367338325259, 0.4965078727255734, 0.7688094663442087, 0.5931780022315986,
0.7766718442305506, 0.8068213064281896, 0.5746132990710054, 0.36734560606074773, 0.8110870168149635, 0.7519934849421102, 0.5389981236588255]
s_ap_mean = [0.84, 0.84, 0.80, 0.84, 0.708, 0.667, 0.68, 0.88, 0.48, 0.60, 0.60, 0.16, 0.44, 0.88, 0.36]
# Set position of bar on X axis
barWidth = 0.1
r1 = np.arange(len(p_ap_mean))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
bands =  ["5v1", "5vB", "2v1", "2vB", "1v1", "1vB", "Bv1", "1v2", "Bv2", "Dv5Murd", "5MurdvD", "1v5", "Bv5", "5Dv5Murd", "5Murdv5D"]
# Make the plot
plt.bar(r1, p_ap_mean, width=barWidth, edgecolor='white', label='Model')
plt.bar(r2, s_ap_mean, width=barWidth, edgecolor='white', label='Human Data')
# Add xticks on the middle of the group bars
#plt.ylim([0.5, 1])
plt.ylabel('Moral permissibility', fontweight='bold')
plt.xlabel('Scenario', fontweight='bold')
plt.title("Moral Permissibility")
plt.xticks([r + 1/3*barWidth for r in range(len(p_ap_mean))], bands, rotation = 30)
plt.legend()
sns.despine()
# plt.subplot(324)
# plt.scatter(x, y, s=80, c=z, marker=(5, 1))
plt.show()
plt.savefig('moral')