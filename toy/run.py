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

odenet_params = []
fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.gca()
fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
y_list = []
inputs = jnp.reshape(jnp.linspace(-2.0, 2.0, 10), (10, 1))
for i in jnp.linspace(0,1,5):
    targets = inputs**3 + i * inputs 
    ode_param = train_ode(inputs,targets)
    odenet_params.append(ode_param)
    print('epsilon',i)
    ax.scatter(inputs, targets, lw=0.5)
    ax.plot(fine_inputs, batched_odenet(ode_param, fine_inputs), label=str(i),lw=0.5)
    y = batched_odenet(ode_param, fine_inputs)
    y_list.append(y)
    
#Plot figure and visualize
ax.legend()
ax.set_xlabel('input')
ax.set_ylabel('output')
plt.legend(('Data predictions', 'ODE Net predictions'))
plt.savefig('data.png')

data = {'y':y_list}
file = open('y_data', 'wb')
# dump information to that file
pickle.dump(data, file)
# close the file
file.close()