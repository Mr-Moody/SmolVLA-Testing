Allow Docker to show windows on your host screen

```bash
xhost +local:docker
```

Create and enter the container

```bash
docker run -it \
  --name franka_noetic \
  --net=host \
  --privileged \
  -v ~/catkin_ws:/home/thomas/catkin_ws \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  osrf/ros:noetic-desktop-full \
  bash
```

```bash
docker exec -it franka noetic bash
```

Before you run the robot code:

Network Check: On your laptop (Host), ensure your ethernet is set to Manual IP 172.16.0.2 and you can ping 172.16.0.1 (the robot).

Web Interface: Open a browser and go to https://172.16.0.1. Log into the Franka Desk and make sure:

The "FCI" (Franka Control Interface) button is blue (Active).

The brakes are unlocked.

The external activation device (the white button) is pressed.

Once that is ready, you will launch the driver inside Docker with:

```bash
source devel/setup.bash
roslaunch franka_control franka_control.launch robot_ip:=172.16.0.1 load_gripper:=true
```