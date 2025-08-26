import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import signal

# Initialize time array
t = np.linspace(0, 2, 1000)

def generate_wave(wave_type, freq, amp, phase, t):
    """Generate a wave based on type and parameters."""
    phase_rad = np.deg2rad(phase)
    if wave_type == 'sine':
        return amp * np.sin(2 * np.pi * freq * t + phase_rad)
    elif wave_type == 'square':
        return amp * signal.square(2 * np.pi * freq * t + phase_rad)
    elif wave_type == 'sawtooth':
        return amp * signal.sawtooth(2 * np.pi * freq * t + phase_rad)
    return np.zeros_like(t)

# Create figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.4)

# Initialize waves
wave1_type = wave2_type = 'sine'
wave1 = generate_wave(wave1_type, freq=1, amp=1, phase=0, t=t)
wave2 = generate_wave(wave2_type, freq=1, amp=1, phase=0, t=t)
superposed = wave1 + wave2

# Plot lines
line1, = ax1.plot(t, wave1, 'b-', lw=2, label='Wave 1')
line2, = ax2.plot(t, wave2, 'r-', lw=2, label='Wave 2')
line3, = ax3.plot(t, superposed, 'g-', lw=2, label='Superposition')

# Axis labels and titles
ax1.set_title('Wave 1')
ax2.set_title('Wave 2')
ax3.set_title('Superposition')
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, 2)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True)

# Create sliders
axcolor = 'lightgoldenrodyellow'
ax_freq1 = plt.axes([0.2, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_freq2 = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_amp1 = plt.axes([0.2, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_amp2 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_phase1 = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_phase2 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

freq1_slider = Slider(ax_freq1, 'Wave 1 Freq', 0.1, 10, valinit=1)
freq2_slider = Slider(ax_freq2, 'Wave 2 Freq', 0.1, 10, valinit=1)
amp1_slider = Slider(ax_amp1, 'Wave 1 Amp', 0.1, 2, valinit=1)
amp2_slider = Slider(ax_amp2, 'Wave 2 Amp', 0.1, 2, valinit=1)
phase1_slider = Slider(ax_phase1, 'Wave 1 Phase', 0, 360, valinit=0)
phase2_slider = Slider(ax_phase2, 'Wave 2 Phase', 0, 360, valinit=0)

# Radio buttons for wave type
rax = plt.axes([0.02, 0.4, 0.1, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('sine', 'square', 'sawtooth'), active=0)

# Reset button
resetax = plt.axes([0.8, 0.02, 0.1, 0.04])
reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def update(val):
    """Update all waves when any slider changes."""
    wave1 = generate_wave(wave1_type, 
                         freq1_slider.val, 
                         amp1_slider.val, 
                         phase1_slider.val, 
                         t)
    wave2 = generate_wave(wave2_type, 
                         freq2_slider.val, 
                         amp2_slider.val, 
                         phase2_slider.val, 
                         t)
    
    line1.set_ydata(wave1)
    line2.set_ydata(wave2)
    line3.set_ydata(wave1 + wave2)
    fig.canvas.draw_idle()

def wave_type_update(label):
    """Update wave type when radio button is clicked."""
    global wave1_type, wave2_type
    wave1_type = wave2_type = label
    update(None)

def reset(event):
    """Reset all sliders to default values."""
    freq1_slider.reset()
    freq2_slider.reset()
    amp1_slider.reset()
    amp2_slider.reset()
    phase1_slider.reset()
    phase2_slider.reset()

# Register update functions
for slider in [freq1_slider, freq2_slider, amp1_slider, amp2_slider, phase1_slider, phase2_slider]:
    slider.on_changed(update)

radio.on_clicked(wave_type_update)
reset_button.on_clicked(reset)

plt.show()