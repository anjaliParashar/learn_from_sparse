import numpy as np
import matplotlib.pyplot as plt

def lane_change_trajectory(v, t_total, lane_width, curvature_factor):
    """
    Generate the trajectory for a lane change maneuver.

    Parameters:
    v (float): Velocity of the vehicle (m/s).
    t_total (float): Total time for lane change (seconds).
    lane_width (float): Width of the lane (meters).
    curvature_factor (float): Factor controlling the curvature. 
                              Higher values result in a sharper turn.

    Returns:
    x (numpy array): x-coordinates of the trajectory.
    y (numpy array): y-coordinates of the trajectory.
    t (numpy array): Time array for the trajectory.
    """
    
    # Time array
    t = np.linspace(0, t_total, num=100)

    # x-coordinates assuming constant velocity
    x = v * t

    # y-coordinates with curvature control
    y = (lane_width / 2) * (1 - np.cos(curvature_factor * np.pi * t / t_total))

    return x, y, t

def plot_trajectory(x, y):
    """
    Plot the lane change trajectory.

    Parameters:
    x (numpy array): x-coordinates of the trajectory.
    y (numpy array): y-coordinates of the trajectory.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Lane Change Trajectory")
    plt.title("Lane Change Trajectory")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # User inputs
    #velocity = 20  # m/s (72 km/h)
    time_for_lane_change = 5  # seconds
    lane_width = 3.5  # meters
    #curvature_factor = 1.5  # Adjust this for different curvature

    Vel = np.linspace(1,20,20)
    Curv = np.linspace(1,5,20)
    for velocity,curvature_factor in zip(Vel,Curv):
        # Generate trajectory   
        x, y, t = lane_change_trajectory(velocity, time_for_lane_change, lane_width, curvature_factor)
        # Plot the trajectory
        plot_trajectory(x, y)
