import pickle
import jax.numpy as jnp

import matplotlib.pyplot as plt
# open a file, where you stored the pickled data
file = open('y_data.pkl', 'rb')

# dump information to that file
data = pickle.load(file)
y_list = jnp.array(data['y']).squeeze()
# close the file
file.close()


fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
extra_inputs = fine_inputs[-10:]
target_list = []
for i in jnp.linspace(0,2,5):
    targets = fine_inputs**3 + i * fine_inputs 
    target_list.append(targets)

target_list = jnp.array(target_list).squeeze()
cov_list = []
input_list = []

i_list = []
for i in range(100):
    cov_i = jnp.cov(target_list[:,i])
    cov_list.append(cov_i)
    if cov_i>4.75:
        input_list.append(fine_inputs[i])
        i_list.append(i)
        

print(len(input_list))
cov_list.sort()
print(cov_list[-10:])
#compute true value y=f(x) for these values of x

epsilon_est = 1.9
gamma = 0.1

#Trial-1
epsilon_est1 = [epsilon_est]
i=0
for input in input_list:
    target_true = input**3 + 0.25 * input 
    target = input**3 + epsilon_est*input
    print('Trial-1',epsilon_est)
    epsilon_est = epsilon_est - gamma*(target-target_true)*input
    epsilon_est1.append(epsilon_est[0])
    if i>=4:
        break
    i+=1

#Trial-2
epsilon_est = 1.9
i=0
epsilon_est2=[epsilon_est]
for input in extra_inputs:
    target_true = input**3 + 0.25 * input 
    target = input**3 + epsilon_est*input
    epsilon_est = epsilon_est - gamma*(target-target_true)*input
    epsilon_est2.append(epsilon_est[0])
    print('Trial-2',epsilon_est)
    if i>=4:
        break
    i+=1
I = jnp.linspace(0,6,6)
plt.plot(I,epsilon_est1,marker='o',label='Experiment-1')
plt.plot(I,epsilon_est2,marker='^', label='Experiment-2')
plt.title('$\\theta$ estimation for 5 datapoints')
plt.ylabel('$\\theta-\\theta_{estimate}$',fontsize=15)
plt.hlines(0.25,0,4,linestyle='--',color='black')
plt.xlabel('Number of data points')
plt.legend()


from jax import random
key = random.PRNGKey(758493)  # Random seed is explicit in JAX
weights = random.uniform(key, shape=(5,1))
y_nn = jnp.array(y_list)
for inputs,i in zip(input_list,i_list):
    f = y_nn[:,i].reshape((5,1))
    y_pred = jnp.sum(jnp.multiply(weights,y_nn[:,i].reshape((5,1))))   
    target = inputs**3 + 0.25 * inputs
    weights = weights - (target-y_pred)*f


