import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import inspect


# ============================================================
# Aperture Shape Functions
# ============================================================


def create_circular_aperture(N, radius_frac=0.2):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    aperture = np.zeros((N, N))
    aperture[R <= radius_frac] = 1
    return aperture

def create_circular_edge_aperture(N, outer_radius_frac=0.3, thickness_frac=0.02):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    aperture = np.zeros((N, N))
    aperture[(R <= outer_radius_frac) & (R >= outer_radius_frac - thickness_frac)] = 1
    return aperture

def create_square_aperture(N, side_frac=0.3):
    aperture = np.zeros((N, N))
    center = N // 2
    half_side = int(side_frac * N / 2)
    aperture[center - half_side : center + half_side, center - half_side : center + half_side] = 1
    return aperture

def create_square_edge_aperture(N, outer_side_frac=0.3, thickness_frac=0.02):
    aperture = np.zeros((N, N))
    center = N // 2
    half_outer = int(outer_side_frac * N / 2)
    half_inner = int((outer_side_frac - thickness_frac) * N / 2)
    aperture[center - half_outer : center + half_outer, center - half_outer : center + half_outer] = 1
    aperture[center - half_inner : center + half_inner, center - half_inner : center + half_inner] = 0
    return aperture

def create_linear_slit(N, width_frac=0.05):
    aperture = np.zeros((N, N))
    center = N // 2
    half_width = int(width_frac * N / 2)
    aperture[:, center - half_width : center + half_width] = 1
    return aperture

def create_rectangular_aperture(N, width_frac=0.1, height_frac=0.3):
    aperture = np.zeros((N, N))
    center = N // 2
    half_w = int(width_frac * N / 2)
    half_h = int(height_frac * N / 2)
    aperture[center - half_h : center + half_h, center - half_w : center + half_w] = 1
    return aperture

def create_cross_aperture(N, arm_width_frac=0.05):
    aperture = np.zeros((N, N))
    center = N // 2
    half_w = int(arm_width_frac * N / 2)
    aperture[:, center - half_w : center + half_w] = 1
    aperture[center - half_w : center + half_w, :] = 1
    return aperture

def create_hexagon_aperture(N, radius_frac=0.3):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    aperture = np.zeros((N, N))
    r = radius_frac
    condition1 = np.abs(X) <= r * np.cos(np.pi / 6)
    condition2 = np.abs(Y) <= r
    condition3 = np.abs(Y) <= np.sqrt(3) * r - np.sqrt(3) * np.abs(X)
    aperture[condition1 & condition2 & condition3] = 1
    return aperture

def create_hexagon_edge_aperture(N, outer_radius_frac=0.3, thickness_frac=0.02):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    r_outer = outer_radius_frac
    r_inner = outer_radius_frac - thickness_frac
    aperture = np.zeros((N, N))

    cond_outer1 = np.abs(X) <= r_outer * np.cos(np.pi / 6)
    cond_outer2 = np.abs(Y) <= r_outer
    cond_outer3 = np.abs(Y) <= np.sqrt(3) * r_outer - np.sqrt(3) * np.abs(X)

    cond_inner1 = np.abs(X) <= r_inner * np.cos(np.pi / 6)
    cond_inner2 = np.abs(Y) <= r_inner
    cond_inner3 = np.abs(Y) <= np.sqrt(3) * r_inner - np.sqrt(3) * np.abs(X)

    outer_hex = cond_outer1 & cond_outer2 & cond_outer3
    inner_hex = cond_inner1 & cond_inner2 & cond_inner3

    aperture[outer_hex] = 1
    aperture[inner_hex] = 0
    return aperture

# ============================================================
# Diffraction Pattern Calculation
# ============================================================

def compute_diffraction_pattern(aperture):
    fft_result = np.fft.fftshift(np.fft.fft2(aperture))
    intensity = np.abs(fft_result) ** 2
    intensity /= np.max(intensity)
    return intensity

# ============================================================
# Viewer Class
# ============================================================

class BaseViewer:
    def __init__(self, aperture, pattern):
        self.aperture = aperture
        self.pattern = pattern
        self.log_scale = True

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.3)

        self.axs[0].imshow(self.aperture, cmap='gray')
        self.axs[0].set_title("Aperture")
        self.axs[0].axis('off')

        self.img_pattern = self.axs[1].imshow(
            np.log10(self.pattern + 1e-8),
            cmap='hot',
            extent=[-1, 1, -1, 1]
        )
        self.axs[1].set_title("Diffraction pattern (log scale)")

        ax_button_toggle = plt.axes([0.25, 0.15, 0.2, 0.075])
        self.button_toggle = Button(ax_button_toggle, 'Toggle log/linear')
        self.button_toggle.on_clicked(self.toggle_scale)

        ax_button_zoom = plt.axes([0.55, 0.15, 0.2, 0.075])
        self.button_zoom = Button(ax_button_zoom, 'Zoom center')
        self.button_zoom.on_clicked(self.zoom_center)

        ax_button_reset = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button_reset = Button(ax_button_reset, 'Reset zoom')
        self.button_reset.on_clicked(self.reset_zoom)

        plt.show()

    def toggle_scale(self, event):
        if self.log_scale:
            self.img_pattern.set_data(self.pattern)
            self.axs[1].set_title("Diffraction pattern (linear scale)")
        else:
            self.img_pattern.set_data(np.log10(self.pattern + 1e-8))
            self.axs[1].set_title("Diffraction pattern (log scale)")
        self.log_scale = not self.log_scale
        self.fig.canvas.draw_idle()

    def zoom_center(self, event):
        zoom_fraction = 0.1
        self.axs[1].set_xlim(-zoom_fraction, zoom_fraction)
        self.axs[1].set_ylim(-zoom_fraction, zoom_fraction)
        self.fig.canvas.draw_idle()

    def reset_zoom(self, event):
        self.axs[1].set_xlim(-1, 1)
        self.axs[1].set_ylim(-1, 1)
        self.fig.canvas.draw_idle()

# ============================================================
# Main
# ============================================================

def main():
    N = 512

    # Automatically collect aperture functions
    aperture_funcs = {name: obj for name, obj in globals().items() 
                      if callable(obj) and name.startswith("create_") and not name.endswith("_edge_aperture")}

    print("\nSelect aperture shape:")
    for i, name in enumerate(aperture_funcs.keys(), 1):
        print(f"{i}: {name}")
    choice = int(input("Enter your choice: "))

    if choice < 1 or choice > len(aperture_funcs):
        print("Invalid choice! Exiting.")
        return

    shape_name = list(aperture_funcs.keys())[choice - 1]
    func = aperture_funcs[shape_name]
    params = inspect.signature(func).parameters

    # Aperture type
    print("\nAperture type:")
    print("1: Open shape (light passes inside)")
    print("2: Block shape (light passes outside)")
    print("3: Edge mask shape (thin edge only)")
    type_choice = int(input("Enter your choice: "))

    # Parameters for shape
    kwargs = {}
    if "radius_frac" in params:
        kwargs["radius_frac"] = 0.3
    if "width_frac" in params:
        kwargs["width_frac"] = 0.05
    if "height_frac" in params:
        kwargs["height_frac"] = 0.3
    if "side_frac" in params:
        kwargs["side_frac"] = 0.3
    if "arm_width_frac" in params:
        kwargs["arm_width_frac"] = 0.05

    # Open or block logic
    if type_choice == 1:
        aperture = func(N, **kwargs)
    elif type_choice == 2:
        aperture = 1 - func(N, **kwargs)
    elif type_choice == 3:
        edge_name = shape_name.replace("_aperture", "_edge_aperture")
        if edge_name in globals():
            edge_func = globals()[edge_name]
            edge_params = inspect.signature(edge_func).parameters
            edge_kwargs = {}

            if "outer_radius_frac" in edge_params:
                edge_kwargs["outer_radius_frac"] = kwargs.get("radius_frac", 0.3)
                edge_kwargs["thickness_frac"] = 0.02
            if "outer_side_frac" in edge_params:
                edge_kwargs["outer_side_frac"] = kwargs.get("side_frac", 0.3)
                edge_kwargs["thickness_frac"] = 0.02

            aperture = edge_func(N, **edge_kwargs)
        else:
            print(f"\n❌ Edge mask version '{edge_name}' not found for this shape. Please implement it first.\n")
            return
    else:
        print("Invalid type choice! Exiting.")
        return

    pattern = compute_diffraction_pattern(aperture)
    BaseViewer(aperture, pattern)

if __name__ == "__main__":
    main()
