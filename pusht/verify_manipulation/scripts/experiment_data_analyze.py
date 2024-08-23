import pickle
import ipdb
import matplotlib.pyplot as plt

file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/experiments/4.pkl','rb')
data = pickle.load(file)
file.close()

print('obs:',data['obs'].shape)
print('reward:',data['reward'])

fig,axs = plt.subplots(1,2)
axs[0].plot(data['obs'][:,2],data['obs'][:,3],marker='o')
axs[0].set_title('T-block')
axs[0].set_ylim([0,600])
axs[0].set_xlim([0,600])

axs[1].plot(data['obs'][:,0],data['obs'][:,1],marker='o')
axs[1].set_title('End-effector')
axs[1].set_ylim([0,600])
axs[1].set_xlim([0,600])
plt.show()