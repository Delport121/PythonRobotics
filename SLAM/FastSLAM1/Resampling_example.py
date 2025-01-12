"""

FastSLAM 1.0 example

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import pathlib
import sys
import copy

import matplotlib.pyplot as plt
import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from utils.angle import angle_mod

# Fast SLAM covariance
Q = np.diag([0.01, np.deg2rad(00.01)]) ** 2 #0 may cause issues with cholesky decomposition
R = np.diag([0.0, np.deg2rad(00.0)]) ** 2
R = np.diag([0.2, np.deg2rad(10.0)]) ** 2 #Determine variance in input for different particles


#  Simulation parameter
Q_SIM = np.diag([0.01, np.deg2rad(0.01)]) ** 2
R_SIM = np.diag([0.00, np.deg2rad(00.0)]) ** 2 #Determine a variance in input that is used for all particles
OFFSET_YAW_RATE_NOISE = 0.0

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

DT = 1  # time tick [s]
SIM_TIME = 10.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
N_PARTICLE = 500  # number of particle
NTH = N_PARTICLE / 1.05  # Number of particle for re-sampling



show_animation = True
global_history = []  # Stores the trajectories of dead particles

save_fig_number = 0



class Particle:

    def __init__(self, n_landmark):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((n_landmark, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((n_landmark * LM_SIZE, LM_SIZE))
        self.history = [( self.x, self.y, self.yaw)]  # Initialize history with the current state


def fast_slam1(particles, u, z, save_fig_number):
    
    prev_particles = copy.deepcopy(particles) # Store the previous particles for plotting propagation lines. 
    #We need to do this type of copy otherwise the particles will be linked and the prev_particles will be updated as well
  
    #First plot t=0
    for i in range(N_PARTICLE):
                plt.plot(particles[i].x, particles[i].y, ".g", markersize = 5)
    # print("Prev particles x", prev_particles[0].x)
    # print("Prev particles y", prev_particles[0].y)
    
    plt.xlim([-0.2, 7.2])
    plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
   
    predicted_particles = predict_particles(particles, u)
    
    # Plot propagation lines
    for i in range(N_PARTICLE):
        plt.plot(
            [prev_particles[i].x, predicted_particles[i].x],
            [prev_particles[i].y, predicted_particles[i].y],
            "-o",color="grey", linewidth=0.5,markersize = 2,zorder=1  # Blue line for propagation
        )
        # plt.plot(predicted_particles[i].x, predicted_particles[i].y, ".k", markersize=2)  # Predicted particles
    
   
        
    plt.xlim([-0.2, 7.2])
    plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    if save_fig_number != 5:
        plt.savefig(f"SLAM/FastSLAM1/Plots/Resampling_{save_fig_number}.png", bbox_inches="tight")  
        save_fig_number += 1  # Increment the counter for the next iteration
   
    # Plot the GPS coordinate
    plt.scatter(z[0], z[1], c='red', s=50, label='GPS Measurement', zorder = 10, marker = '*')
    plt.xlim([-0.2, 7.2])
    plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    if save_fig_number != 5:
                plt.savefig(f"SLAM/FastSLAM1/Plots/Resampling_{save_fig_number}.png", bbox_inches="tight")  
                save_fig_number += 1  # Increment the counter for the next iteration
    
    # for i in range(N_PARTICLE):
    #         plt.plot(predicted_particles[i].x, predicted_particles[i].y, ".k", markersize = 2)
    
    updated_particles = update_with_observation(predicted_particles, z)
    
    # Plot original particles with color intensity based on weights
    plot_particles_with_weights(updated_particles, z, title="Original Particles Before Resampling", overlay=True)
    plt.xlim([-0.2, 7.2])
    plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    if save_fig_number != 5:
            plt.savefig(f"SLAM/FastSLAM1/Plots/Resampling_{save_fig_number}.png", bbox_inches="tight")  
            save_fig_number += 1  # Increment the counter for the next iteration
    
    resampled_particles = resampling(updated_particles, z)
    
    # for i in range(N_PARTICLE):
    #             plt.plot(resampled_particles[i].x, resampled_particles[i].y, ".r", markersize = 5)
                
     
    plt.show()

    return predicted_particles, resampled_particles


def normalize_weight(particles):
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles

def calc_final_state(particles):
    x_est = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        x_est[0, 0] += particles[i].w * particles[i].x
        x_est[1, 0] += particles[i].w * particles[i].y
        x_est[2, 0] += particles[i].w * particles[i].yaw

    x_est[2, 0] = pi_2_pi(x_est[2, 0])

    return x_est


def predict_particles(particles, u):
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        ud = u + (np.random.randn(1, 2) @ R ** 0.5).T  # add noise
        px = motion_model(px, ud)
        particles[i].x = px[0, 0]
        particles[i].y = px[1, 0]
        particles[i].yaw = px[2, 0]

        # particles[i].history.append((particles[i].x, particles[i].y, particles[i].yaw))

    return particles



def update_with_observation(particles, z_gps, std_dev=0.08):
    """
    Update particles' weights based on the GPS measurement.
    """
    gps_position = z_gps  # GPS coordinates [x, y]

    for ip in range(N_PARTICLE):
        # Compute the Euclidean distance between the particle and the GPS position
        distance = np.linalg.norm([particles[ip].x - gps_position[0], particles[ip].y - gps_position[1]])
        # Assign a higher weight to particles closer to the GPS position
        particles[ip].w = (np.exp(-0.5 * (distance / std_dev)**2))

      
    pw = np.array([p.w for p in particles])
    # print("Weights before normalising:", pw)

    return particles


def resampling(particles, gps_coordinate):
    """
    Low variance re-sampling with visualization of the resampling process.
    """
    particles = normalize_weight(particles)



    # Collect weights
    pw = np.array([p.w for p in particles])
    
    # MIN_WEIGHT = 1e-1  # Adjust this based on your use case
    # pw = np.array([p.w for p in particles])
    # pw[pw < MIN_WEIGHT] = 0.0
    # pw /= np.sum(pw)



    # Effective number of particles
    n_eff = 1.0 / np.sum(pw ** 2)
    print("Eff Threshold:", NTH)
    print(f"Effective particle number (n_eff): {n_eff}")

    if n_eff < NTH:  # Resampling condition
        print("Resampling")
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        indexes = []
        index = 0
        for ip in range(N_PARTICLE):
            while (index < w_cum.shape[0] - 1) and (resample_id[ip] > w_cum[index]):
                index += 1
            indexes.append(index)

        
        
        
        # print("Weights normalised:", pw)
        # print("Cumulative Weights:", w_cum)
        # print("Resample IDs:", resample_id)
        print(f"Resampling indexes: {indexes}")

        tmp_particles = particles[:]
        for i in range(len(indexes)):
            particles[i].x = tmp_particles[indexes[i]].x
            particles[i].y = tmp_particles[indexes[i]].y
            particles[i].yaw = tmp_particles[indexes[i]].yaw
            particles[i].lm = tmp_particles[indexes[i]].lm[:, :]
            particles[i].lmP = tmp_particles[indexes[i]].lmP[:, :]
            particles[i].w = 1.0 / N_PARTICLE

    # Plot resampled particles on top of the original particles
    # plot_particles_with_weights(particles, gps_coordinate, title="Resampled Particles", overlay=True)
   
    return particles


def calc_input(time):
    # if time <= 3.0:  # wait at first
    #     v = 0.0
    #     yaw_rate = 0.0
    # else:
    #     v = 1.0  # [m/s]
    #     yaw_rate = 0.0  # [rad/s]

    v = 3.0  # [m/s]
    yaw_rate = 0.0  # [rad/s]
    u = np.array([v, yaw_rate]).reshape(2, 1)

    return u


def observation(x_true, xd, u, rfid):
    # calc true state
    x_true = motion_model(x_true, u)

    # add noise to range observation
    z = np.zeros((3, 0))
    for i in range(len(rfid[:, 0])):

        dx = rfid[i, 0] - x_true[0, 0]
        dy = rfid[i, 1] - x_true[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - x_true[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_SIM[0, 0] ** 0.5  # add noise
            angle_with_noize = angle + np.random.randn() * Q_SIM[
                1, 1] ** 0.5  # add noise
            zi = np.array([dn, pi_2_pi(angle_with_noize), i]).reshape(3, 1)
            z = np.hstack((z, zi))

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_SIM[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_SIM[
        1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return x_true, z, xd, ud

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    v = u[0].item() #+ np.random.normal(0, v_noise_std)
    omega = u[1].item() #+ np.random.normal(0, omega_noise_std)

    if abs(omega) > 1e-3:  # Avoid division by zero for very small angular velocity
        dx = v / omega * (math.sin(x[2, 0] + omega * DT) - math.sin(x[2, 0]))
        dy = v / omega * (-math.cos(x[2, 0] + omega * DT) + math.cos(x[2, 0]))
    else:
        dx = v * DT * math.cos(x[2, 0])
        dy = v * DT * math.sin(x[2, 0])

    x[0, 0] += dx
    x[1, 0] += dy
    x[2, 0] += omega * DT
    x[2, 0] = pi_2_pi(x[2, 0])  # Normalize angle

    return x



def pi_2_pi(angle):
    return angle_mod(angle)

# def plot_particles_with_weights(particles, z):
#     """
#     Plot particles, with color intensity corresponding to the weight.
#     """
#     x_vals = [particle.x for particle in particles]
#     y_vals = [particle.y for particle in particles]
#     weights = [particle.w for particle in particles]

#     # Normalize weights to map to a color scale
#     norm_weights = np.array(weights) / max(weights)

#     # Scatter plot of particles, color-coded by normalized weight
#     plt.plot(z[0], z[1], "xr")
#     plt.scatter(x_vals, y_vals, c=norm_weights, cmap='viridis', s=10)
#     plt.colorbar(label='Weight')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Particle Positions with Weights')
#     plt.show()

def plot_particles_with_weights(particles, gps_coordinate, title, overlay=False):
    """
    Plot particles, with color intensity corresponding to the weight.
    Optionally overlay particles on an existing plot.
    """
    x_vals = [particle.x for particle in particles]
    y_vals = [particle.y for particle in particles]
    weights = [particle.w for particle in particles]

    # Normalize weights to map to a color scale
    norm_weights = np.array(weights) / max(weights)

    if not overlay:
        plt.figure()

    # Scatter plot of particles, color-coded by normalized weight
    scatter = plt.scatter(x_vals, y_vals, c=norm_weights, cmap='coolwarm', s=10, label='Particles') #cmap='viridis'

    # Plot the GPS coordinate
    plt.scatter(gps_coordinate[0], gps_coordinate[1], c='red', s=50, label='GPS Measurement', zorder = 10, marker = '*')

    # Add colorbar only for the first plot
    if not overlay:
        plt.colorbar(scatter, label='Weight')

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title(title)
    # plt.legend()
    # plt.axis("equal")
    # plt.grid(True)

    if not overlay:
        plt.show()


def main():
    print(__file__ + " start!!")

    time = 0.0
    save_fig_number = 0

    # RFID positions [x, y]
    rfid = np.array([[1.5, -1.],
                     [1.5, 1.],
                     ])
  
    n_landmark = rfid.shape[0]

    # State Vector [x y yaw v]'
    x_est = np.zeros((STATE_SIZE, 1))  # SLAM estimation
    x_true = np.zeros((STATE_SIZE, 1))  # True state
    x_dr = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # History
    hist_x_est = x_est
    hist_x_true = x_true
    hist_x_dr = x_dr

    # Initialize particles
    particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]

    # #First plot t=0
    # for i in range(N_PARTICLE):
    #             plt.plot(particles[i].x, particles[i].y, ".g", markersize = 10)
    # plt.plot(rfid[:, 0], rfid[:, 1], "*k")
    # plt.show()
    # if save_fig_number != 10:
    #             plt.savefig(f"SLAM/FastSLAM1/Plots/FastSLAM1_{save_fig_number}.png", bbox_inches="tight")  
    #             save_fig_number += 1  # Increment the counter for the next iteration
    
    hist_particles = particles

    while SIM_TIME >= time:
        time += DT
        u = calc_input(time)
    
        ud1 = u[0, 0] + np.random.randn() * R_SIM[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R_SIM[1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
        ud = np.array([ud1, ud2]).reshape(2, 1)

        # Gps measurement
        z = np.array([3,0])
        print(z)

        predicted_particles, particles = fast_slam1(particles, ud, z, save_fig_number)

        x_est = calc_final_state(particles)

        x_state = x_est[0: STATE_SIZE]

        # Store data history
        hist_x_est = np.hstack((hist_x_est, x_state))
        hist_x_dr = np.hstack((hist_x_dr, x_dr))

        if show_animation:  # pragma: no cover
            # plt.cla()
            # # For stopping simulation with the ESC key.
            # plt.gcf().canvas.mpl_connect(
            #     'key_release_event', lambda event:
            #     [exit(0) if event.key == 'escape' else None])

            # # Plot RFID positions
            # # plt.plot(rfid[:, 0], rfid[:, 1], "*k")

            # # Plot particle trajectories
            # for i in range(N_PARTICLE):
            #     plt.plot(predicted_particles[i].x, predicted_particles[i].y, ".", color = "blue", markersize = 6)
            #     plt.plot(particles[i].x, particles[i].y, ".r",  markersize = 3)
            #     # plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            # # Plot SLAM, dead reckoning, and true paths
            # # plt.plot(hist_x_true[0, :], hist_x_true[1, :], "-b")
            # # plt.plot(hist_x_dr[0, :], hist_x_dr[1, :], "-k")
            # # plt.plot(hist_x_est[0, :], hist_x_est[1, :], "-r")
            # plt.plot(x_est[0], x_est[1], "xk")

            # plt.axis("equal")
            # plt.grid(True)
            plt.pause(0.001)

            # if save_fig_number != 10:
            #     plt.savefig(f"SLAM/FastSLAM1/Plots/FastSLAM1_{save_fig_number}.png", bbox_inches="tight")  
            #     save_fig_number += 1  # Increment the counter for the next iteration

if __name__ == '__main__':
    main()


    