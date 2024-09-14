# FIRST Pose Estimation

This is a standalone repo for running experiments with pose estimation for FIRST Robotics games.

## Building
You will need mamba (recommended) or conda, available through [miniforge](https://github.com/conda-forge/miniforge). You will also need:
* cmake
* libgtk2.0-dev
and a C++ compiler. If you are on a Debian flavor, you can get these all through:
```
sudo apt install build-essential cmake libgtk2.0-dev
```
After this, create the environment:
```
mamba env create -f environment.yml
mamba activate first
```
And the build the dependencies:
```
# build Eigen
cd dependencies/eigen
mkdir -p build && cd build
cmake .. && make -j8
cd ../../..
# build the Python packages
cd dependencies/apriltag
python -m pip install .
cd ../..
cd dependencies/PoseLib
python -m pip install .
cd ../..
```

## Running
For now, running is very simple:
```
python3 estimate.py
```
This will try to run pose estimation against all JPGs in the `data` folder and will draw green quadrangles around all the detected AprilTags in the image. Press any key to go to the next one.