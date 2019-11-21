#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
matplotlib.use('Agg')

from matplotlib import pylab as plt
import matplotlib.backends.backend_agg as agg

DPI = 96
RESOLUTION = 800, 600
SUPPORTED_MODES = {
    'human': lambda: DeepRmHumanRenderer,
    'rgb_array': lambda: DeepRmRgbRenderer
}
pygame = None


class DeepRmRgbRenderer(object):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        self.resolution = resolution
        self.dpi = DPI

    @staticmethod
    def plot_substate(ax, title, state, colorbar=False):
        im = ax.imshow(state, cmap='rainbow', vmin=0, vmax=1)
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

        procs, mem, wait_procs, wait_mem, backlog = state
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


class DeepRmHumanRenderer(DeepRmRgbRenderer):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        super().__init__(resolution, dpi)

        global pygame
        if pygame is None:
            import pygame

        pygame.init()
        self.screen = pygame.display.set_mode(resolution, pygame.DOUBLEBUF)
        pygame.display.set_caption('Scheduler State')

    def render(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                raise SystemExit

        rgb, size = super().render(state)
        surface = pygame.image.fromstring(rgb, size, "RGB")
        surface_position = surface.get_rect()
        self.screen.blit(surface, surface_position)
        pygame.display.update()

        return rgb, size


class DeepRmRenderer(object):
    def __init__(self, mode, *args, **kwargs):
        if mode not in SUPPORTED_MODES:
            raise RuntimeError('Requested unsupported mode %s' % mode)
        self.renderer = SUPPORTED_MODES[mode]()(*args, **kwargs)

    def render(self, state):
        return self.renderer.render(state)