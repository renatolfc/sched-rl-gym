#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np

import pyglet
from matplotlib import pylab as plt
import matplotlib.backends.backend_agg as agg

DPI = 96
WIDTH = 800
HEIGHT = 600
RESOLUTION = (WIDTH, HEIGHT)

SUPPORTED_MODES = {
    'human': lambda: DeepRmHumanRenderer,
    'rgb_array': lambda: DeepRmRgbRenderer,
}


class DeepRmRgbRenderer(object):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        self.resolution = resolution
        self.dpi = DPI

    @staticmethod
    def plot_substate(ax, title, state, colorbar=False):
        cmap = matplotlib.cm.get_cmap('rainbow')
        cmap.set_under('w')
        im = ax.imshow(state, cmap=cmap, vmin=0.001, vmax=1)
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Slots')
        ax.set_ylabel('Time horizon (timesteps)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid()

    def render(self, state):
        width = self.resolution[0] / self.dpi
        height = self.resolution[1] / self.dpi
        fig = plt.figure(0, figsize=(width, height), dpi=self.dpi)

        current, wait, backlog, time = state
        lines = current.shape[0]

        # Axes {{{
        axs_current = [
            plt.subplot2grid((lines, 3), (i, 0)) for i in range(lines)
        ]
        axs_wait = [plt.subplot2grid((lines, 3), (i, 1)) for i in range(lines)]
        ax_backlog = plt.subplot2grid((lines, 3), (0, 2), rowspan=lines)
        # End of Axes }}}

        for i, (ax_current, ax_wait) in enumerate(zip(axs_current, axs_wait)):
            self.plot_substate(
                ax_current, f'Current resources {i}', current[i]
            )
            self.plot_substate(
                ax_wait, f'Waiting jobs stack {i}', np.mean(wait[i], axis=0)
            )
        self.plot_substate(ax_backlog, 'Backlog', backlog, True)

        fig.tight_layout()
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        plt.close(fig)

        return np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (size[0], size[1], 3)
        )


class DeepRmHumanRenderer(DeepRmRgbRenderer, pyglet.window.Window):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        super().__init__(resolution, dpi)

        self.rendering = None
        width, height = resolution
        self.window = pyglet.window.Window(width, height, visible=False)
        self.window.set_caption('Scheduler State')
        self.window.set_visible()
        self.window.on_draw = self.on_draw

    def on_draw(self):
        self.window.clear()
        if self.rendering is not None:
            height, width, _ = self.rendering.shape
            img = pyglet.image.ImageData(
                height,
                width,
                'RGB',
                self.rendering.data.tobytes(),
                -3 * height,
            )

            img.blit(0, 0)

    def render(self, state):
        self.rendering = super().render(state)

        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()

        return self.rendering


class DeepRmRenderer(object):
    def __init__(self, mode, *args, **kwargs):
        if mode not in SUPPORTED_MODES:
            raise RuntimeError('Requested unsupported mode %s' % mode)
        self.renderer = SUPPORTED_MODES[mode]()(*args, **kwargs)

    def render(self, state):
        return self.renderer.render(state)
