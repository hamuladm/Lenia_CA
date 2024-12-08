"""
Visualization module for Lenia using Pygame
"""

import pygame
import numpy as np
from lenia import Lenia


class LeniaVisualizer:
    """
    Visualizing for LeniaVisualizer class using Pygame
    """

    def __init__(self, lenia: Lenia, screen_width: int = 512, screen_height: int = 512):
        """
        Initialization Visu class \

        It initializes all the necessary attributes
        """
        self.lenia = lenia
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

    @staticmethod
    def to_screen_size(matrix: np.ndarray, screen_width: int, screen_height: int):
        """
        Makes every pixel a square(using numpy)
        """
        matrix = matrix.copy()
        matrix = matrix.transpose(1, 0, 2)
        matrix = np.flip(matrix, 0)
        matrix = np.rot90(matrix)
        matrix = np.repeat(matrix, screen_width // matrix.shape[0], axis=0)
        matrix = np.repeat(matrix, screen_height // matrix.shape[1], axis=1)
        return matrix

    def __draw(self):
        pygame.display.set_caption("Lenia")
        self.screen.fill((0, 0, 0))

        # Convert torch tensors to numpy
        world = [
            c.cpu().numpy() for c in self.lenia.world
        ]  # Makes '3D-array for RGB visualization'
        world = np.dstack(world)  # Makes '3D-array for RGB visualization'
        world *= 255  # Scalar multiplying
        world = self.to_screen_size(world, self.screen_width, self.screen_height)

        surf = pygame.surfarray.make_surface(world)
        self.screen.blit(surf, (0, 0))

        pygame.display.update()

    def run(self):
        """
        Main loop for the visualization
        """
        while True:
            self.__draw()
            self.lenia.next()
            self.clock.tick(100)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()


if __name__ == "__main__":
    import sys

    PATTERN = sys.argv[1] if len(sys.argv) > 1 else "aquarium"

    _lenia = Lenia(
        pattern=PATTERN, size=512, scale=4, start_x=150, start_y=100, cells_amount=4
    )
    visualizer = LeniaVisualizer(_lenia, screen_height=512, screen_width=512)
    visualizer.run()
