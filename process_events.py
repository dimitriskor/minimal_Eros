import event_processing
import numpy as np
import cv2

# Initialize EROS surface
eros = event_processing.EROS()
eros.init(640, 480, kernel_size=5, parameter=0.9)

# Generate a batch of events: (x, y, timestamp, polarity)
num_events = 10000  # Simulating many events
events = np.random.randint(0, 640, (num_events, 4)).astype(np.float64)
events[:, 1] = np.random.randint(0, 480, num_events)  # y values
events[:, 2] = np.random.uniform(0, 1, num_events)    # timestamps
events[:, 3] = np.random.choice([-1, 1], num_events)  # polarity

# Pass all events to C++ at once (fast!)
eros.update_batch(events)

# Apply decays
eros.temporal_decay(0.03, -0.1)
eros.spatial_decay(3)

# Get and display the result
surface = eros.get_surface()
cv2.imshow("EROS Surface", surface)
cv2.waitKey(0)
cv2.destroyAllWindows()