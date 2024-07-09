import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
from jax.experimental.ode import odeint
from jax import vmap
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/anjali/learn_from_sparse/toy')
from train import train_ode, batched_odenet
import pickle

file = 'y_data.pkl'

file_i = open(file, 'rb')
# dump information to that file
data = pickle.load(file_i)
file_i.close()

y_list = data['y']
fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))

fig = plt.figure(figsize=(6, 4), dpi=150)
inputs = jnp.reshape(jnp.linspace(-2.0, 2.0, 10), (10, 1))

ax = fig.gca()
i_idx=0
for i in jnp.linspace(0,1,5):
    targets = fine_inputs**3 + i * fine_inputs 
    ax.scatter(fine_inputs, targets, lw=0.5)
    ax.plot(fine_inputs, y_list[i_idx], lw=0.5,label=str(i))
    i_idx+=1
ax.legend()
ax.set_xlabel('input')
ax.set_ylabel('output')
plt.legend()
#plt.savefig('data.png')