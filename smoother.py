import numpy as np

class Smoother:
    def __init__(self, buffer_size=5):
        """
        buffer_size: how many frames to average over.
        Higher = smoother but slightly more lag.
        5 is a good starting point.
        """
        self.buffer_size = buffer_size
        self.x_buffer = []
        self.y_buffer = []

    def smooth(self, x, y):
        """Add a new point and return the smoothed coordinates."""
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        # Keep buffer at max size
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)

        smoothed_x = int(np.mean(self.x_buffer))
        smoothed_y = int(np.mean(self.y_buffer))

        return smoothed_x, smoothed_y

    def reset(self):
        """Clear the buffer - useful when hand re-enters frame."""
        self.x_buffer = []
        self.y_buffer = []