import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
p_ap_mean = [0.277908761, 0.275800102, 0.280426084, 0.9399871, 0.926234784, 0.924426234]
s_ap_mean = [0.2, 0.16, 0.08, 1, 0.48, 0.44]
# Set position of bar on X axis
barWidth = 0.1
r1 = np.arange(len(p_ap_mean))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
bands =  ["5vB", "2vB", "1vB", "Bv1", "Bv2", "Bv5"]
# Make the plot
plt.bar(r1, p_ap_mean, width=barWidth, edgecolor='white', label='Model')
plt.bar(r2, s_ap_mean, width=barWidth, edgecolor='white', label='Human Data')
# Add xticks on the middle of the group bars
#plt.ylim([0.5, 1])
plt.ylabel('Norm = only loved ones are valued', fontweight='bold')
plt.xlabel('Scenario', fontweight='bold')
plt.title("Likelihood of agent believing in norm given the action of pulling the switch")
plt.xticks([r + 1/3*barWidth for r in range(len(p_ap_mean))], bands, rotation = 30)
plt.legend()
sns.despine()
# plt.subplot(324)
# plt.scatter(x, y, s=80, c=z, marker=(5, 1))
plt.show()
plt.savefig('norm')