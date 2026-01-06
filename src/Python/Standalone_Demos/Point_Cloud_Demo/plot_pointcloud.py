import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "pointcloud.csv"

def plot_pointcloud():
    df = pd.read_csv(CSV_FILE)

    if df.empty:
        print("CSV is empty — run radar_parse_pointcloud.py first.")
        return

    # Top-down view: x vs y, color by velocity
    plt.figure()
    scatter = plt.scatter(df['x_m'], df['y_m'], c=df['vel_mps'], s=6)
    plt.colorbar(scatter, label="Velocity (m/s)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters — distance from radar)")
    plt.title("Radar Point Cloud (Top-Down View)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    plot_pointcloud()
