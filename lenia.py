"""
Lenia module
"""

import torch
from scipy import ndimage
from patterns import PATTERNS


class Lenia:
    """
    Lenia model
    """

    patterns = PATTERNS

    def __init__(
        self,
        pattern: str = "orbium",
        size: int = 64,
        scale: int = 1,
        start_x: int = 20,
        start_y: int = 20,
        cells_amount: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialization function for Lenia class

        It initializes all the necessary attributes
        """
        assert 1 <= cells_amount <= 5, "Wrong number of cells"

        self.device = device
        self.pattern = self.patterns[pattern]  # What pattern it will use for generating
        self.size = size  # Size of field
        self.scale = scale
        self.start_pos = (start_x, start_y)  # Where it starts
        self.world = [
            torch.zeros((size, size), device=self.device) for _ in range(3)
        ]  # Full black world

        self.__init_pattern()
        self.period = self.pattern["T"]
        self.fourier_kernels = self.__smooth_ring_kernel(self.pattern["R"] * self.scale)

    def __init_pattern(self):
        """
        Initialize the world with the pattern
        """
        cells = self.pattern["cells"]
        cells = [ndimage.zoom(c, self.scale, order=0) for c in cells]
        for world, cells_group in zip(self.world, cells):
            cells_group = torch.tensor(cells_group, device=self.device)
            world[
                self.start_pos[0] : self.start_pos[0] + cells_group.shape[0],
                self.start_pos[1] : self.start_pos[1] + cells_group.shape[1],
            ] = cells_group

    @staticmethod
    @torch.jit.script
    def bell_function(value: torch.Tensor, c_mu: float, c_sigma: float) -> torch.Tensor:
        """
        Bell function.
        Used for calculating growth values.
        """
        return torch.exp(-(((value - c_mu) / c_sigma) ** 2) / 2)

    @staticmethod
    def soft_clip(value: torch.Tensor) -> torch.Tensor:
        """
        Soft clip function.
        Returns value x: 0 <= x <= 1
        """
        return 1 / (1 + torch.exp(-4 * (value - 0.5)))

    def __smooth_ring_kernel(self, radius: float) -> list[torch.Tensor]:
        mid = self.size // 2

        # Create distance matrix using torch
        y, x = torch.meshgrid(
            torch.arange(-mid, mid, device=self.device, dtype=torch.float),
            torch.arange(-mid, mid, device=self.device, dtype=torch.float),
            indexing="ij",
        )
        distance_array = [
            torch.sqrt(x**2 + y**2) / radius * len(k["b"]) / k["r"]
            for k in self.pattern["kernels"]
        ]

        kernels = []
        for dmatrix, k in zip(distance_array, self.pattern["kernels"]):
            # Create kernel with torch operations
            mask = dmatrix < len(k["b"])
            indices = torch.minimum(
                dmatrix.to(torch.long),
                torch.tensor(len(k["b"]) - 1, device=self.device),
            )
            kernel_base = torch.tensor(k["b"], device=self.device)[indices]
            kernel = (
                mask.float() * kernel_base * self.bell_function(dmatrix % 1, 0.5, 0.15)
            )
            kernels.append(kernel)

        # Normalize kernels
        normalized_kernels = [kernel / kernel.sum() for kernel in kernels]

        # Compute Fourier transforms
        fourier_kernels = [
            torch.fft.fftshift(nkernel).to(torch.complex64)
            for nkernel in normalized_kernels
        ]
        fourier_kernels = [torch.fft.fft2(kernel) for kernel in fourier_kernels]

        return fourier_kernels

    def __growth(self, matrix: torch.Tensor, c_mu: float, c_sigma: float):
        return self.bell_function(matrix, c_mu, c_sigma) * 2 - 1

    def __update(self):
        # Compute Fourier transform of world channels
        fourier_world = torch.stack(
            [torch.fft.fft2(world_channel) for world_channel in self.world]
        )

        # Vectorized convolution and growth calculations
        wn_sums = torch.fft.ifft2(
            torch.stack(self.fourier_kernels)
            * fourier_world[[k["c0"] for k in self.pattern["kernels"]]]
        ).real
        # Compute growth
        growns = [
            self.__growth(wn_sum, k["m"], k["s"])
            for wn_sum, k in zip(wn_sums, self.pattern["kernels"])
        ]

        # Compute growth heights
        growth_heights = [
            sum(
                k["h"] * grow
                for grow, k in zip(growns, self.pattern["kernels"])
                if k["c1"] == c1
            )
            for c1 in range(3)
        ]

        # Update world
        self.world = [
            self.soft_clip(world_channel + 1 / self.period * g_height)
            for world_channel, g_height in zip(self.world, growth_heights)
        ]

    def next(self):
        """
        Updates the world and return it
        """
        self.__update()
        return self.world
