import matplotlib.pyplot as plt
import numpy as np


class Arena(object):
    """Arena for agents that move, look, and see coloured walls.

    color_function -- a function accepting array-likes of angles in [0, 2pi) and
    returning the corresponding colours of locations on the circular wall.
    visual_breadth -- the angle describing the width of the agent's visual field
    visual_resolution -- how many colour samples can the agent's eyes see
    """
    def __init__(self, color_function,
                 visual_breadth, visual_resolution):
        self.colorfunc = color_function
        self.visual_res = visual_resolution
        self.visual_breadth = visual_breadth
        self._pos = (0.0, 0.0)
        self._phi = 0.0
        self.update_vf()

    def _get_rays(self):
        rng = np.linspace(-self.visual_breadth, self.visual_breadth,
                          self.visual_res)
        return (self._phi + rng)

    def _get_circle_intersection(self, ray):
        x0, y0 = self._pos
        sgn = np.sign(np.cos(ray))
        m = np.tan(ray)
        if np.abs(m) > 1e10:
            solY = np.sign(np.sin(ray)) * np.sqrt(1 - x0**2)
            return x0, solY
        A = 1 + m**2
        Bhalf = (m * y0 - m**2 * x0)
        C = y0**2 - 2 * m * y0 * x0 + m**2 * x0**2 - 1
        solX = (-Bhalf + sgn*np.sqrt(Bhalf**2 - A*C))/A
        solY = y0 + m*(solX - x0)
        return solX, solY

    def update_vf(self):
        """Updates the visual field (vf_xy, vf_angles, vf_colors) reflecting
        changes in other variables."""
        rays = self._phi + np.linspace(
            -self.visual_breadth, self.visual_breadth, self.visual_res)
        inters = np.empty([len(rays), 2])
        for i, r in enumerate(rays):
            inters[i] = self._get_circle_intersection(r)
        self.vf_xy = inters

        vf_angles = np.arctan(inters[:, 1]/inters[:, 0]) + np.pi*(inters[:, 0] < 0)
        self.vf_angles = (vf_angles[::-1] % (2*np.pi))

        self.vf_colors = self.colorfunc(self.vf_angles)

    @property
    def pos(self):
        """The location of the agent, in x and y coordinates."""
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        """The location of the agent, in x and y coordinates.
        The visual field is automatically updated after setting."""
        x, y = new_pos
        if x**2 + y**2 > 1:
            raise ValueError("Position outside unit circle")
        self._pos = new_pos
        self.update_vf()

    @property
    def phi(self):
        """The orientation of the agent, in [0, 2pi)."""
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        """The orientation of the agent, in [0, 2pi).
        The visual field is automatically updated after setting."""
        self._phi = new_phi % (2*np.pi)
        self.update_vf()

    def plot_arena(self, ax=None, walls=100):
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        circ = plt.Circle((0, 0), radius=.99, facecolor='#ffffee', edgecolor='k')
        ax.add_patch(circ)

        for inters in self.vf_xy:
            ax.plot([self.pos[0], inters[0]],
                    [self.pos[1], inters[1]],
                    'k', linewidth=.2)
        ax.plot(*self.pos, '*r')

        if isinstance(walls, int):
            wvector = np.linspace(0, 2*np.pi, walls)
            wcolors = self.colorfunc(wvector)
            ax.scatter(np.cos(wvector), np.sin(wvector), c=wcolors)

    def plot_visual_field(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Agent's current visual field")

        ax.scatter(np.arange(self.visual_res),
                   np.zeros(self.visual_res),
                   c=self.vf_colors,
                   marker='s')

    def plot_combined(self):
        plt.figure(figsize=(5, 7))
        plt.ylim([-1.4, 1.2])
        plt.annotate("Arena view", xy=(-.19, 1.1))
        self.plot_arena()
        plt.annotate("Agent's visual field", xy=(-.375, -1.2))
        plt.scatter(np.linspace(-1, 1, self.visual_res),
                    -1.3 * np.ones(self.visual_res),
                    c=self.vf_colors,
                    marker='s')
