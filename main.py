import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONSTANTE ---
R_EARTH = 6371e3            # raza Pământului (m)
GM = 3.986004418e14         # G * M_earth (m^3/s^2)
# (opțional, poți folosi 3.9860047e14)

# --- PARAMETRI SIMULARE ---
t_max = 10000.0  # durata simulării în secunde (~2.7 ore)
dt = 1.0         # pas de ieșire în fișiere (nu neapărat pas de integrare)

# --- CONDIȚII INIȚIALE ---
# Lansăm de la (R_EARTH, 0) (pe ecuator), cu o viteză inițială orientată sub un unghi "theta0" față de orizontala locală.
# Viteza inițială (magnitudine):
v0 = 6000.0      # m/s (sub-orbital; < ~7.9 km/s necesar pentru orbită circulară)
# Unghi de lansare față de orizontala locală (grade):
launch_angle_deg = 45.0
launch_angle = np.radians(launch_angle_deg)

# Coordonate inițiale:
x0 = R_EARTH
y0 = 0.0

# Viteza inițială: orizontală + verticală (față de orizontală)
vx0 = v0 * np.cos(launch_angle)
vy0 = v0 * np.sin(launch_angle)

# Vectorul stare inițial: [x, y, vx, vy]
y_init = [x0, y0, vx0, vy0]

# --- FUNCȚIA DE DERIVATE ---
def rocket_equations(t, state):
    """
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    # accelerațiile
    ax = -GM * x / (r**3)
    ay = -GM * y / (r**3)
    return [vx, vy, ax, ay]

# --- INTEGRAREA ---
# Vom integra până la t_max, dar dacă racheta intră în Pământ (r < R_EARTH),
# putem opri simularea.
def stop_if_below_surface(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    return r - R_EARTH  # semn: devine negativ când r < R_EARTH

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
r_vals = np.sqrt(x_vals**2 + y_vals**2)           # distanța față de centrul Pământului
h_vals = r_vals - R_EARTH                         # altitudine
v_vals = np.sqrt(vx_vals**2 + vy_vals**2)         # viteza scalară

# Downrange distance: unghiul = arctan2(y, x), raportat la unghiul inițial
theta_vals = np.arctan2(y_vals, x_vals)
theta0 = theta_vals[0]
downrange_vals = R_EARTH * np.abs(theta_vals - theta0)

# Flight-path angle (față de orizontală).
#   - vector radial: (x, y)
#   - vector viteză: (vx, vy)
#   - unghiul dintre viteza rachetei și radială: 
#       cos(phi) = (v . r) / (|v| |r|)
#   - flight_path_angle = 90° - phi  (fiind unghiul cu orizontala locală)
flight_path_angles = []
for i in range(len(t_vals)):
    # radial
    rx, ry = x_vals[i], y_vals[i]
    rr = r_vals[i]
    # viteză
    vx, vy = vx_vals[i], vy_vals[i]
    vv = v_vals[i]
    # unghiul dintre v și r
    dot_v_r = rx*vx + ry*vy
    cos_phi = dot_v_r / (rr * vv + 1e-12)  # adăugăm un epsilon să evităm /0
    # Încadrare cos_phi între [-1, 1], din cauza erorilor numerice
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)  # unghi cu radiala
    # flight-path angle cu orizontala = 90° - phi
    gamma = np.pi/2 - phi
    flight_path_angles.append(np.degrees(gamma))

flight_path_angles = np.array(flight_path_angles)

# --- 3. GRAFICE ---
fig = plt.figure(figsize=(12, 10))
plt.suptitle("Simulare simplificată a traiectoriei (2D, doar gravitație)", fontsize=14)

# 1) Height vs Time
ax1 = plt.subplot(3, 2, 1)
ax1.plot(t_vals, h_vals/1000, label="Altitudine")
ax1.set_xlabel("Timp [s]")
ax1.set_ylabel("Altitudine [km]")
ax1.set_title("Height vs. Time")
ax1.grid(True)

# 2) Velocity vs Time
ax2 = plt.subplot(3, 2, 2)
ax2.plot(t_vals, v_vals, label="Viteză", color='g')
ax2.set_xlabel("Timp [s]")
ax2.set_ylabel("Viteză [m/s]")
ax2.set_title("Velocity vs. Time")
ax2.grid(True)

# 3) Flight-path angle vs Time
ax3 = plt.subplot(3, 2, 3)
ax3.plot(t_vals, flight_path_angles, label="Unghi traiectorie", color='r')
ax3.set_xlabel("Timp [s]")
ax3.set_ylabel("Unghi (° față de orizontala locală)")
ax3.set_title("Flight-Path Angle vs. Time")
ax3.grid(True)

# 4) Downrange distance vs Time
ax4 = plt.subplot(3, 2, 4)
ax4.plot(t_vals, downrange_vals/1000, label="Dist. orizontală", color='m')
ax4.set_xlabel("Timp [s]")
ax4.set_ylabel("Downrange [km]")
ax4.set_title("Downrange Distance vs. Time")
ax4.grid(True)

# 5) Height vs. Downrange distance (2D plot) - traiectorie în plan (x, y) proiectată pe planul orizontal (x, z) 
ax5 = plt.subplot(3, 2, 5)
ax5.plot(downrange_vals/1000, h_vals/1000, color='c')
ax5.set_xlabel("Downrange [km]")
ax5.set_ylabel("Altitudine [km]")
ax5.set_title("Height vs. Downrange")
ax5.grid(True)

# 6) Vedere polar / top-down
#    racheta în coordonate polare (r, theta), însă pentru un aspect „plan” desenăm un cerc cu raza Pământului
ax6 = plt.subplot(3, 2, 6, projection='polar')
# Pământul (cerc de rază R_EARTH în km)
earth_radius_km = R_EARTH/1000
theta_circle = np.linspace(0, 2*np.pi, 200)
ax6.plot(theta_circle, [earth_radius_km]*len(theta_circle), 'b', label="Pământul")

# Traiectoria rachetei
# Convertim r_vals din m în km, theta_vals sunt deja unghiuri
r_km = r_vals/1000
ax6.plot(theta_vals, r_km, 'r', label="Traiectoria")
ax6.set_rmax(r_km.max()*1.1)
ax6.set_theta_zero_location("E")   # 0° la est (unde x>0, y=0)
ax6.set_theta_direction(-1)        # sens trigonometric invers (pentru a simula "sus" = nord)
ax6.set_title("Traiectorie văzută de sus (polar)")
ax6.legend(loc="upper right")

plt.tight_layout()
plt.show()
