import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

filename = '../Datasets/Artificial/risk.pdf'

fig, ax1 = plt.subplots(frameon=False)
rc('mathtext', default='regular')
rc('lines',lw=1.6)
rc('lines',mew=1.4)
rc('text', usetex=True)

x = np.arange(0.,2.1,0.2)

rs = -(1-np.exp(x))
rn = x
ra = 1-np.exp(-x)

lns1 = ax1.plot(x,ra,'ks:', markersize=3, label='Risk Aversion');
plt.text(1.85,0.5,'(a=1)',color='k')
lns2 = ax1.plot(x,rn,'cs-.', markersize=3, label="Risk Neutral");
plt.text(1.85,1.5,'(a=0)',color='c')
lns3 = ax1.plot(x,rs,'bs-',markersize=3, label='Risk Seeking');
plt.text(1.85,5,'(a=-1)',color='b')

ax1.set_ylabel(r'$g(x)$',size=13);
ax1.set_xlabel(r'$x$',size=13);
ax1.set_xlim([0.,2.1]);

lns = lns1 + lns2 + lns3;
labs = [l.get_label() for l in lns];

lgd = ax1.legend(lns, labs, bbox_to_anchor=(0.5,0.5),loc=4, ncol = 1, fontsize='11',shadow=True,fancybox=True);
fig.savefig(filename,format='pdf',transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight');
