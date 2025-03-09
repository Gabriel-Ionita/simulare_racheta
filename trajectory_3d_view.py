import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# --- CONSTANTE ---
R_EARTH = 6371e3            # raza Pământului (m)
GM = 3.986004418e14         # G * M_earth (m^3/s^2)

# --- PARAMETRI SIMULARE ---
t_max = 10000.0  # durata simulării în secunde (~2.7 ore)
dt = 1.0         # pas de ieșire în fișiere (nu neapărat pas de integrare)

# --- CONDIȚII INIȚIALE ---
v0 = 6000.0      # m/s (sub-orbital; < ~7.9 km/s necesar pentru orbită circulară)
launch_angle_deg = 45.0
launch_angle = np.radians(launch_angle_deg)

x0 = R_EARTH
y0 = 0.0

vx0 = v0 * np.cos(launch_angle)
vy0 = v0 * np.sin(launch_angle)

y_init = [x0, y0, vx0, vy0]

# --- FUNCȚIA DE DERIVATE ---
def rocket_equations(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    ax = -GM * x / (r**3)
    ay = -GM * y / (r**3)
    return [vx, vy, ax, ay]

# --- INTEGRAREA ---
def stop_if_below_surface(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    return r - R_EARTH

stop_if_below_surface.terminal = True
stop_if_below_surface.direction = -1

sol = solve_ivp(
    rocket_equations, 
    [0, t_max], 
    y_init, 
    t_eval=np.arange(0, t_max, dt),
    events=stop_if_below_surface,
    rtol=1e-8, 
    atol=1e-8
)

# Extragem rezultatele
t_vals = sol.t
x_vals = sol.y[0]
y_vals = sol.y[1]
vx_vals = sol.y[2]
vy_vals = sol.y[3]

# --- CALCULE COMPLEMENTARE PENTRU GRAFICE ---
r_vals = np.sqrt(x_vals**2 + y_vals**2)
h_vals = r_vals - R_EARTH
v_vals = np.sqrt(vx_vals**2 + vy_vals**2)

theta_vals = np.arctan2(y_vals, x_vals)
theta0 = theta_vals[0]
downrange_vals = R_EARTH * np.abs(theta_vals - theta0)

flight_path_angles = []
for i in range(len(t_vals)):
    rx, ry = x_vals[i], y_vals[i]
    rr = r_vals[i]
    vx, vy = vx_vals[i], vy_vals[i]
    vv = v_vals[i]
    dot_v_r = rx*vx + ry*vy
    cos_phi = dot_v_r / (rr * vv + 1e-12)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    gamma = np.pi/2 - phi
    flight_path_angles.append(np.degrees(gamma))

flight_path_angles = np.array(flight_path_angles)

# --- GRAFIC 3D ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Pământul (sferă de rază R_EARTH în km)
earth_radius_km = R_EARTH / 1000
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius_km * np.outer(np.cos(u), np.sin(v))
y = earth_radius_km * np.outer(np.sin(u), np.sin(v))
z = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Traiectoria rachetei
x_km = x_vals / 1000
y_km = y_vals / 1000
z_km = h_vals / 1000
ax.plot(x_km, y_km, z_km, 'r', label="Traiectoria")

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Altitudine [km]')
ax.set_title("Traiectorie văzută în 3D")
ax.legend(loc="upper right")

plt.show()