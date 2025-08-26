import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import gcd, pi

# Lissajous Curve Parameters
A = 1.0          # X-amplitude
B = 1.0          # Y-amplitude
p = 3            # X-frequency numerator
q = 2            # Y-frequency numerator
delta = pi/4     # Phase shift (radians)
resolution = 1000  # Points in curve

# Calculate reduced ratio
g = gcd(p, q)
p_reduced = p // g
q_reduced = q // g

# Calculate period for closed curve (rational frequency ratio)
T = 2 * pi * max(p_reduced, q_reduced)  # Time period for one full pattern

# Create time array
t = np.linspace(0, T, resolution)

# Parametric equations
x = A * np.sin(p * t + delta)
y = B * np.sin(q * t)

# Create figure
fig, (ax, ax_phase) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle(f'Lissajous Pattern: ωx/ωy = {p}:{q}, δ = {delta:.2f} rad', fontsize=16)

# Main Lissajous plot
ax.plot(x, y, 'b-', linewidth=1.5)
ax.set_xlim(-A*1.1, A*1.1)
ax.set_ylim(-B*1.1, B*1.1)
ax.set_xlabel('x = A sin(ωx t + δ)')
ax.set_ylabel('y = B sin(ωy t)')
ax.set_title('Lissajous Curve')
ax.grid(True)
ax.set_aspect('equal', 'box')

# Draw amplitude boundaries
ax.plot([-A, A], [-B, -B], 'r--', alpha=0.5)
ax.plot([-A, A], [B, B], 'r--', alpha=0.5)
ax.plot([-A, -A], [-B, B], 'r--', alpha=0.5)
ax.plot([A, A], [-B, B], 'r--', alpha=0.5)

# Phase space visualization
time_point = 0
phase = np.linspace(0, 2*pi, 100)
ax_phase.plot(A * np.sin(phase), B * np.sin(phase), 'g-', alpha=0.3)
ax_phase.plot(A * np.sin(phase), B * np.cos(phase), 'm-', alpha=0.3)
phase_line1, = ax_phase.plot([0, A * np.sin(p * time_point + delta)], 
                             [0, B * np.sin(q * time_point)], 'bo-')
phase_line2, = ax_phase.plot([0, A * np.sin(p * time_point + delta)], 
                             [0, B * np.cos(q * time_point)], 'mo-', alpha=0.5)
ax_phase.set_xlim(-A*1.1, A*1.1)
ax_phase.set_ylim(-B*1.1, B*1.1)
ax_phase.set_xlabel('X-component')
ax_phase.set_ylabel('Y-component')
ax_phase.set_title('Phase Space Visualization')
ax_phase.grid(True)
ax_phase.set_aspect('equal', 'box')
ax_phase.legend(['sin-sin', 'sin-cos'])

# Add physics explanations
physics_text = (
    f"Physics Insights:\n"
    f"• Frequency ratio: {p}:{q} {'(rational)' if p/q == p_reduced/q_reduced else '(irrational)'}\n"
    f"• Pattern closed: {'Yes' if p/q == p_reduced/q_reduced else 'No'}\n"
    f"• Nodes: {p_reduced + q_reduced} visible\n"
    f"• Area: {pi*A*B*abs(np.sin(delta)):.2f} (proportional to |sinδ|)\n"
    f"• Mathematical form:\n"
    f"  x = {A}·sin({p}t + {delta:.2f})\n"
    f"  y = {B}·sin({q}t)"
)
fig.text(0.05, 0.05, physics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Add special cases information
special_cases = {
    (1, 1, 0): "Straight line: y = (B/A)x",
    (1, 1, pi/2): "Ellipse" + (" (Circle)" if A==B else ""),
    (2, 1, pi/4): "Parabolic approximation",
    (3, 2, 0): "Complex figure-8 pattern"
}

current_case = None
for params, desc in special_cases.items():
    if abs(p - params[0]) < 0.01 and abs(q - params[1]) < 0.01 and abs(delta - params[2]) < 0.01:
        current_case = desc

if current_case:
    ax.text(0, -B*0.9, f"Special Case: {current_case}", 
            ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

# Animation function
point, = ax.plot([], [], 'ro', markersize=8)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    point.set_data([], [])
    time_text.set_text('')
    return point, time_text

def update(frame):
    # Update moving point
    point.set_data(x[frame], y[frame])
    time_text.set_text(f't = {t[frame]:.2f} s\n'
                       f'Phase x: {(p*t[frame] + delta) % (2*pi):.2f} rad\n'
                       f'Phase y: {(q*t[frame]) % (2*pi):.2f} rad')
    
    # Update phase space visualization
    phase_line1.set_data([0, A * np.sin(p * t[frame] + delta)], 
                         [0, B * np.sin(q * t[frame])])
    phase_line2.set_data([0, A * np.sin(p * t[frame] + delta)], 
                         [0, B * np.cos(q * t[frame])])
    
    return point, time_text, phase_line1, phase_line2

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, 
                    interval=20, blit=True)

# Save slow-motion MP4
ani.save(
    'lissajous_slow.mp4',
    writer='ffmpeg',
    fps=10,    # Lower fps → slower playback
    dpi=200,
    bitrate=1800
)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()
