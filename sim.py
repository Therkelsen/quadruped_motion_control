import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -5)

# Load the floor
planeId = p.loadURDF("plane.urdf")

# Load your robot
startPos = [0, 0, 0.5]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("a1_description/urdf/a1.urdf", startPos, startOrientation)

num_joints = p.getNumJoints(robot)
controllable_joints = []
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    if info[2] == p.JOINT_REVOLUTE:
        controllable_joints.append(i)
        
print("Controllable joints:", controllable_joints)

j = 0
h = 0
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./360)
    j += 1
    if j % 500 == 0:
        if h < len(controllable_joints):
            print("moving ", h)
            p.setJointMotorControl2(robot, controllable_joints[h], p.POSITION_CONTROL, targetPosition=0.2)
            h += 1
        
p.disconnect()
