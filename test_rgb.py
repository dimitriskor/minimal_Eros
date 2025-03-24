import tqdm
import sys
sys.path.append("./build")
import event_processing
import numpy as np
import cv2
import os
import h5py
import hdf5plugin
import cv2
import matplotlib.pyplot as plt

prefix = 'event_data/event_recordings/'
folders = os.listdir(prefix)
# Initialize EROS surface
kernel_size = 9
delta_1 = 0.25 ** (1/kernel_size)
delta_2 = 0.5 ** (1/kernel_size)
delta_3 = 0.75 ** (1/kernel_size)

for folder in folders:
    filename = prefix + folder + '/events/left/events.h5'
    file = h5py.File(filename)
    events = file['events']
    ms_to_idx = file['ms_to_idx']
    iterations = len(ms_to_idx)//50-1

    eros_1 = event_processing.EROS()
    eros_2 = event_processing.EROS()
    eros_3 = event_processing.EROS()
    eros_1.init(640, 480, kernel_size, delta_1)
    eros_2.init(640, 480, kernel_size, delta_2)
    eros_3.init(640, 480, kernel_size, delta_3)
    frames = []
    for i in tqdm.tqdm(range(iterations), total=iterations):
        
        start, stop = ms_to_idx[i*50], ms_to_idx[(i+1)*50]
        p, t, x, y = events['p'][start:stop], events['t'][start:stop], events['x'][start:stop], events['y'][start:stop]
        x = np.array(x, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        t = np.array(t, dtype=np.int64)  # or dtype=np.float64 if t should be float
        p = np.array(p, dtype=np.int32)   # if p is supposed to be a boolean

        batch = np.stack([x, y, t, p], axis=1, dtype=np.float64)
        eros_1.update_batch(batch)
        eros_2.update_batch(batch)
        eros_3.update_batch(batch)
        surface_1 = eros_1.get_surface()
        surface_2 = eros_2.get_surface()
        surface_3 = eros_3.get_surface()
        # print((surface_1 - surface_2).sum())
        frames.append(np.array((surface_1, surface_2, surface_3)).transpose(1, 2, 0))
        plt.imshow(frames[-1])
        plt.savefig('sdsdfbsfdb.png')
        # print(i,'/',iterations,end='\r')
        # csdgdf

    try:
        os.makedirs(f'frames_rgb/{folder}')
    except:
        pass
    for idx, frame in enumerate(frames):
        image_name = f'frames_rgb/{folder}/{idx:05}.png'
        cv2.imwrite(image_name, frame)
    
    del eros_1
    del eros_2
    del eros_3



# # Generate a batch of events: (x, y, timestamp, polarity)
# num_events = 10000000  # Simulating many events
# events = np.random.randint(0, 640, (num_events, 4)).astype(np.float64)
# events[:, 1] = np.random.randint(0, 480, num_events)  # y values
# events[:, 2] = np.random.uniform(0, 1, num_events)    # timestamps
# events[:, 3] = np.random.choice([-1, 1], num_events)  # polarity

# # Pass all events to C++ at once (fast!)
# print(events.shape)

# # Apply decays
# eros.temporal_decay(0.03, -0.1)
# eros.spatial_decay(3)

# Get and display the result
# cv2.imshow("EROS Surface", surface)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

