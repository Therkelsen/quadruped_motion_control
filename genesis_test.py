import time
import torch
import genesis as gs

# --------------------------------------------------------------------
# Initialize Genesis
# --------------------------------------------------------------------
try:
    gs.init(backend=gs.gpu)
except Exception:
    print("⚠️ GPU backend not available — falling back to CPU.")
    gs.init(backend=gs.cpu)


# --------------------------------------------------------------------
# Minimal Go2 Scene Setup (no training)
# --------------------------------------------------------------------
class Go2Scene:
    def __init__(self, show_viewer=True, device="cuda"):
        self.device = torch.device(device)
        self.dt = 0.02  # 50 Hz physics step

        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=10,
                gravity=[0, 0, -9.81]
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt),
                camera_pos=(2.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True
            ),
            show_viewer=show_viewer
        )

        # Add a ground plane
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True
            )
        )

        # Add the Go2 robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=[0.0, 0.0, 0.6],  # start slightly above the ground
                quat=[0.0, 0.0, 0.0, 1.0]
            )
        )

        # Build the scene
        self.scene.build(n_envs=1)

        print("✅ Scene initialized with gravity and Go2 robot.")

    def run(self):
        print("🚀 Running simulation (press Ctrl+C to stop)...")
        try:
            while True:
                self.scene.step()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("🛑 Simulation stopped.")
        finally:
            self.scene.destroy()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = Go2Scene(show_viewer=True, device=device)
    sim.run()
