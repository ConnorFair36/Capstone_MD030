# fusion_system.py
#
# Runs BOTH sensors live (radar + RealSense) and prints fused results.
# Shows camera window. Radar window is not displayed (data only).
#
# Fusion rule:
#   - If camera sees a person → use CAMERA distance + RADAR motion
#   - If camera sees no person but radar sees motion → occluded target
#   - If neither sees anything → no person detected

import time
import radar_module
import camera_module

def main():
    print("=== Initializing sensors ===")
    radar_module.initialize()
    camera_module.initialize()
    print("=== Sensors ready. Running fusion ===")
    print("Press 'q' on the camera window to exit.\n")

    while True:
        # ---- Radar update ----
        radar_data = radar_module.get_radar_data()
        # radar_data is either None or { "distance": float | None, "motion": str }

        radar_dist  = None
        radar_motion = "NONE"

        if radar_data is not None:
            radar_dist = radar_data.get("distance", None)
            radar_motion = radar_data.get("motion", "NONE")

        # ---- Camera update ----
        cam_data = camera_module.get_camera_data()
        if cam_data == "QUIT":
            break

        # cam_data is { "person": bool, "distance": float | None }
        cam_person   = cam_data["person"]
        cam_distance = cam_data["distance"]

        # ---- Fusion logic ----
        if not cam_person and radar_motion == "NONE":
            msg = "No person detected"

        elif cam_person:
            # camera has priority for distance
            msg = f"Person detected — {cam_distance:.2f} m — {radar_motion}"

        elif not cam_person and radar_motion != "NONE":
            msg = f"Possible hidden target — motion detected — radar distance {radar_dist:.2f} m"

        else:
            msg = "Unknown state"

        print(msg)
        time.sleep(0.01)  # small delay for print readability

    print("Exiting fusion system.")

if __name__ == "__main__":
    main()
