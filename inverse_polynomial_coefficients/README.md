# Inverse Polynomial Computation for Cartesian Polynomial Model (CPM)

## Overview
This script computes the inverse polynomial transformation for a camera using the Cartesian Polynomial Model (CPM). The output is a JSON file containing the computed inverse polynomial coefficients and relevant camera parameters. This JSON file can then be used with [py-oamcalib](https://github.com/jakarto3d/py-OCamCalib/) to undistort images based on the CPM model.

This work is inspired by [py-ocamcalib](https://github.com/jakarto3d/py-OCamCalib/), which is a Python adaptation of the original MATLAB [OcamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab) by Scaramuzza et al.

## Installation
Clone this repository:
```sh
git clone git@github.com:isri-aist/Utils.git
cd Utils/inverse_polynomial_coefficients
```
Ensure you have Python 3 and the required dependencies installed:
```sh
pip install numpy
```

## Usage
To compute the inverse polynomial, run the following command:
```sh
python3 inverse.py --parameters <path_to_parameters_file>
```
This will generate a file named `*_inverse.json` with the inverse polynomial coefficients.

## Input Format
The input parameter file should be in the following format:
```
Camera:
  name: "PanoraMIS Catadioptric"
  model: "CPM"
  
  a0: 117.073
  a1: 0
  a2: -0.002
  a3: 0
  a4: 0
  
  u0: 321.364
  v0: 311.875
  
  cols: 620
  rows: 620
```

## Output Format
The generated JSON file will have the following structure:
```json
{
    "date": "18052022_154907",
    "camera_name": "panoramis",
    "valid": [true, true, true, false, false, false, false, false, false, false, false, false, false],
    "taylor_coefficient": [117.073, 0.0, -0.002, 0.0, 0.0],
    "distortion_center": [321.364, 311.875],
    "stretch_matrix": [[1, 0], [0, 1]],
    "inverse_poly": [0.353, -7.618, 72.206, -386.476, 1206.483, -1671.068, -2976.070, 22065.534, -61083.924, 107740.760, -133499.720, 119548.533, -77546.546, 35928.714, -11543.915, 2449.386, -304.910, 21.602, 116.501, 0.001]
}
```

## References
- **Scaramuzza, D., Martinelli, A., & Siegwart, R. (2006).** *A Toolbox for Easily Calibrating Omnidirectional Cameras*. Optical Express, 14(21), 8576-8589. [Link](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)
- **[py-ocamcalib](https://github.com/jakarto3d/py-OCamCalib/)** (GitHub)



