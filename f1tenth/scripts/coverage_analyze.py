import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
Z_xr = [2.0,2.0,2.0,3.0,1.0]
Z_yr = [0.0,-1.0,1.0,0.0,0.0]
Z_list =[]
dispersion_list = []
reward_list = []
for Z_x,Z_y in zip(Z_xr, Z_yr):
    print(Z_x,Z_y)
    file = open("/home/anjali/learn_from_sparse/f1tenth/data/MH_circle_proj/MH_"+str(Z_x)+ "_"+str(Z_y)+".pkl", 'rb')
    # dump information to that file
    data = pickle.load(file)
    file.close()
    dispersion_list+=data['disp']
    reward_list +=data['risk']
    Z_list+=data['Z']
    

Z_np = np.array(Z_list).squeeze()
dispersion_np = np.array(dispersion_list).squeeze()

plt.scatter(Z_np[:,0],Z_np[:,1])
plt.show()

plt.figure()
plt.plot(dispersion_np,'o')
plt.show()

#Analysis of results
N_idx = np.where(dispersion_np>1.0)
print(N_idx)
print('Z:',Z_np[N_idx,:])
print('Disperson:',dispersion_np[N_idx])
Z_plt = Z_np[N_idx,:].squeeze()
plt.scatter(Z_plt[:,0],Z_plt[:,1])
plt.title('Initial sampling zone')
plt.show()
#plt.plot(reward_list[N_idx],'o')

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,**kwargs))
    plt.show()
        

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    N = gmm.means_.shape[0]
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.7 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
#K-mean clustering on the generated data
gmm = GaussianMixture(n_components=20).fit(Z_plt)
kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(Z_plt)
vor = Voronoi(Z_plt)
fig = voronoi_plot_2d(vor)
plt.show()

labels = gmm.predict(Z_plt)
labels1 = kmeans.predict(Z_plt)
plt.figure()
plt.scatter(Z_plt[:, 0], Z_plt[:, 1], c=labels1, s=40, cmap='viridis')
plt.title('GMM clustering')
plt.show()

plt.figure()
Z_means1 = kmeans.cluster_centers_
Z_means = gmm.means_
plt.scatter(Z_means1[:,0],Z_means1[:,1])
plt.title('Means of N/2 clusters')
plt.show()

experiment_list = []
for i in range(20):
    dist_ = cdist(Z_plt,Z_means1[i,:].reshape((1,2)))
    idx = np.argmin(dist_)
    experiment_list.append(Z_plt[idx,:])
    print(reward_list[idx])

Z_exp = np.array(experiment_list)
#Pack means into a pickle file
data={'means':Z_means1,'data':Z_plt,"Z_exp":Z_exp}
file = open("/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/experiment_initial_1.pkl", 'wb')
pickle.dump(data, file)
file.close()
#gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
#plot_gmm(gmm, Z_plt)
