from collections import deque


class BasePipelineState:
    """Base class for pipeline state."""


class BasePipeline:
    """Base class for pipeline."""

    def __init__(self, buffer_size: int, seed: int):
        if buffer_size <= 0:
            raise ValueError("`buffer_size` must be greater than 0.")
        self.seed = seed
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.buffer_index = -1
        self.shift = 0

    def fetch(self):
        """Return next items to be appended to the buffer."""

        raise NotImplementedError

    def get_fetch_state(self):
        raise NotImplementedError

    def fill_buffer(self):
        """Fill buffer with items that are consumed by the data gateway."""

        while len(self.buffer) < self.buffer_size:
            self.fetch()

    def set_state(self, state: BasePipelineState):
        """Setup the pipeline at an appropriate state ready for continuous data generation."""

        raise NotImplementedError

    def __next__(self):
        self.fill_buffer()
        items = []
        for _ in range(self.buffer_size - self.shift):
            items.append(self.buffer.popleft())
        self.shift = 0
        return items
