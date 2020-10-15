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
    'rgb_array': lambda: DeepRmRgbRenderer
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
        ax.set_xticks(np.arange(-.5, state.shape[1], 1))
        ax.set_yticks(np.arange(-.5, state.shape[0], 1))
        ax.set_xticklabels(range(state.shape[1] + 1), rotation=45, ha='center')
        ax.set_yticklabels(range(state.shape[0] + 1))
        ax.set_title(title)
        ax.set_xlabel('Slots')
        ax.set_ylabel('Time horizon (timesteps)')
        ax.grid()

    def render(self, state):
        width = self.resolution[0] / self.dpi
        height = self.resolution[1] / self.dpi
        fig = plt.figure(0, figsize=(width, height), dpi=self.dpi)

        # Axes {{{
        ax_proc = plt.subplot2grid((2, 3), (0, 0))
        ax_mem = plt.subplot2grid((2, 3), (1, 0))
        ax_wait_proc = plt.subplot2grid((2, 3), (0, 1))
        ax_wait_mem = plt.subplot2grid((2, 3), (1, 1))
        ax_backlog = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
        # End of Axes }}}

        procs, mem, wait_procs, wait_mem, backlog, _ = state
        self.plot_substate(ax_proc, 'Cluster Processors', procs)
        self.plot_substate(ax_mem, 'Cluster Memory', mem)
        self.plot_substate(ax_wait_proc, 'Waiting Processor Stack',
                           np.mean(wait_procs, axis=0))
        self.plot_substate(ax_wait_mem, 'Waiting Memory Stack',
                           np.mean(wait_mem, axis=0))
        self.plot_substate(ax_backlog, 'Backlog', backlog, True)

        fig.tight_layout()
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        plt.close(fig)

        return raw_data, size


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
            (width, height), data = self.rendering
            img = pyglet.image.ImageData(width, height, 'RGB', data)
            # "Fix" projection between OpenGL and matplotlib {{{
            pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
            pyglet.gl.glLoadIdentity()
            pyglet.gl.glOrtho(0.0, width, height, 0.0, -1.0, 1.0)
            pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
            pyglet.gl.glLoadIdentity()
            # }}}

            img.blit(0, 0)


    def render(self, state):
        rgb, size = super().render(state)
        self.rendering = size, rgb

        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()

        return rgb, size


class DeepRmRenderer(object):
    def __init__(self, mode, *args, **kwargs):
        if mode not in SUPPORTED_MODES:
            raise RuntimeError('Requested unsupported mode %s' % mode)
        self.renderer = SUPPORTED_MODES[mode]()(*args, **kwargs)

    def render(self, state):
        return self.renderer.render(state)
