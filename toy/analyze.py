import pickle
import jax.numpy as jnp
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

for i in range(100):
    cov_i = jnp.cov(target_list[:,i])
    cov_list.append(cov_i)
    if cov_i>4.75:
        input_list.append(fine_inputs[i])

print(len(input_list))
cov_list.sort()
print(cov_list[-10:])
#compute true value y=f(x) for these values of x

epsilon_est = 1.9
gamma = 0.1

#Trial-1
for input in input_list:
    target_true = input**3 + 0.25 * input 
    target = input**3 + epsilon_est*input
    print('Trial-1',epsilon_est)
    epsilon_est = epsilon_est - gamma*(target-target_true)*input

#Trial-2
epsilon_est = 1.9
i=0
for input in extra_inputs:
    if i>=5:
        break
    target_true = input**3 + 0.25 * input 
    target = input**3 + epsilon_est*input
    epsilon_est = epsilon_est - gamma*(target-target_true)*input
    print('Trial-2',epsilon_est)
    i+=1



