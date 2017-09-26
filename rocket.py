import matplotlib.pyplot as plt
import numpy as np

# Physical Constants
Fg = -9.8
# Initial Conditions
x_0 = 0
y_0 = 0
dx_0 = 5
dy_0 = 500

# Noise Covariance
Q = 0.15 # process noise covariance
R = 12.2 # measurement noise covariance

t = np.array(range(1000),dtype=np.float)/10

# Positions if model were perfect
x_t_ideal = t * dx_0 + x_0
y_t_ideal = 0.5 * Fg * t**2 + dy_0 * t + y_0

# Actual positions
# Process noise is modeled as random pertubations to projectile
# velocity
x_t_real = [x_0]
y_t_real = [y_0]
dx_real = dx_0
dy_real = dy_0
for i in range(1, len(t)):
  dt = t[i] - t[i-1]
  dx_real = dx_real + np.random.normal(0, Q)
  x_t_real.append(x_t_real[-1] + dt * dx_real)
  dy_real = Fg * dt + dy_real + np.random.normal(0, Q)
  y_t_real.append(y_t_real[-1] + dt * dy_real)

# Noisy measurements
mx_t = x_t_real + np.random.normal(0, R, np.size(x_t_real))
my_t = y_t_real + np.random.normal(0, R, np.size(y_t_real))


# Kalman filter matrices
# These can be constants for this application
# while time interval is constant
dt = t[1] - t[0]
A = np.matrix([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = np.matrix([[1,0,0,0],[0,1,0,0]])
B = np.transpose(np.matrix([[0,0,0,Fg*dt]]))

# assume we know the perfect starting position
x_hat = np.transpose(np.matrix([[x_0, y_0, dx_0, dy_0]]))
x_fil = [x_hat[0,0]]
y_fil = [x_hat[1,0]]
P_k = np.eye(4) # what do I initialize this to? Who knows?
det_P = []
det_P.append(np.linalg.det(P_k))
for i in range(1, len(t)):
  #print(i)
  #print("x_hat: \n{}".format(x_hat))
  dt = t[i] - t[i-1]
  x_hat_prev = A*x_hat + B
  #print("x_hat_prev: \n{}".format(x_hat_prev))
  P_k_prev = A*P_k*np.transpose(A) + np.eye(4) * Q
  K_k = P_k_prev * np.transpose(H) * np.linalg.inv(H * P_k_prev * np.transpose(H) + np.eye(2) * R)
  x_hat = x_hat + K_k * (np.transpose([[mx_t[i],my_t[i]]]) - H * x_hat)
  x_fil.append(x_hat[0,0])
  y_fil.append(x_hat[1,0])
  P_k = (np.eye(4) - K_k * H) * P_k_prev
  det_P.append(np.linalg.det(np.transpose(K_k)*K_k))
fig, ax_list = plt.subplots(2,2)
ax_list[0,0].plot(x_t_real, y_t_real, mx_t, my_t)
ax_list[0,0].set_title("Measurements")
ax_list[0,1].plot(x_t_real, y_t_real, x_t_ideal, y_t_ideal)
ax_list[0,1].set_title("Model Prediction")
ax_list[1,0].plot(x_t_real, y_t_real, x_fil, y_fil)
ax_list[1,0].set_title("Filter Output")
ax_list[1,1].plot(t,det_P)
ax_list[1,1].set_title("Determinant of aposteriori error covariance")

prediction_error = sum(np.sqrt((x_t_ideal - x_t_real)**2 + (y_t_ideal - y_t_real)**2))

measure_error = sum(np.sqrt((mx_t - x_t_real)**2 + (my_t - y_t_real)**2))

filter_error = sum(np.sqrt((np.array(x_fil) - x_t_real)**2 + (np.array(y_fil) - y_t_real)**2))

print("pred {}".format(prediction_error))
print("meas {}".format(measure_error))
print("filter {}".format(filter_error))

plt.show()
