import numpy as np
import pickle
from matplotlib import pyplot as plt


file ="/Users/hadden/Papers/10_chain_dynamics/03_data/TOI-178/TOI-178_three_body_angles_fmft_0_1000.pkl"
with open(file,'rb') as fi:
    fmft_results = pickle.load(fi)

def select_by_nearest_key(d,val):
    keys_arr = np.array(list(d.keys()))
    nearest_key = keys_arr[np.argmin(np.abs(keys_arr - val))]
    return nearest_key,d[nearest_key]

pos_freqs_inner_angle = np.array([np.sort(list(result[0].keys())) for result in  fmft_results])[:,3:]
target_freqs = np.mean(pos_freqs_inner_angle,axis=0)

amp_phi_i = np.zeros((len(target_freqs),len(fmft_results),3),dtype=np.complex128)
freq_i = np.zeros((len(target_freqs),len(fmft_results),3))
for j,result in enumerate(fmft_results):
    for i in range(3):
        for k,f in enumerate(target_freqs):
            freq_i[k,j,i],amp_phi_i[k,j,i] = select_by_nearest_key(result[i],f)


fig,ax = plt.subplots(1,2,figsize = (7,3))
for k in [0,1,2]:
    msk = np.logical_and(np.abs(freq_i[k,:,1]/freq_i[k,:,0]-1)<1e-3,np.abs(freq_i[k,:,2]/freq_i[k,:,0]-1)<1e-3)
    ax[1].scatter(np.abs(amp_phi_i[k,msk]).T[1]/np.abs(amp_phi_i[k,msk]).T[0],np.abs(amp_phi_i[k,msk]).T[2]/np.abs(amp_phi_i[k,msk]).T[0])
ax[1].scatter(1.29748864,0.34839452,s=400,marker='*',color='r')
ax[1].scatter(0.09833555, 0.67352669,s=400,marker='*',color='r')
ax[1].scatter(0.24008978, 0.11923683,s=400,marker='*',color='r')
ax[1].set_xlabel("$A_2/A_1$")
ax[1].set_ylabel("$A_3/A_1$")
ax[1].set_ylim(0,1.4)
ax[1].set_xlim(0,1.4)
bins = np.linspace(0,25,60)
for k in (0,1,2):
    ax[0].hist(np.abs(amp_phi_i[k,:,0]),bins=bins,histtype='step',lw=3)
ax[0].set_xlabel("$A_1$ [deg.]",fontsize=16)
plt.tick_params(direction='in')
ax[0].set_ylabel("$N$",fontsize = 16)
plt.tight_layout()
plt.show()