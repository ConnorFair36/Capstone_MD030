# test_radar_raw.py
import time
import matplotlib.pyplot as plt
from radar_module import initialize, get_radar_data_raw
import numpy as np
from radar_pc_to_camera import radar_to_cam, extrinsic

plt.ion()
fig, ax = plt.subplots()

initialize()

while True:
    points = get_radar_data_raw()

    ax.clear()
    ax.set_title(f"Radar raw points: {len(points)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    ax.grid(True)

    if points.size > 0:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys)

        print("Frame:")
        #for p in points[:5]:
        #    print(f"  x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}, v={p[3]:.2f}")
        #print(points.shape)
        transformed_pc = radar_to_cam(points.T, extrinsic)
        print(transformed_pc)

    plt.pause(0.05)
    time.sleep(0.05)