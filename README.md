# Utilities for Camera Calibration and Image Processing

This repository contains a collection of utilities for camera calibration, image undistortion, and image comparison. The provided tools are organized into different directories, each focusing on a specific functionality.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Available Tools](#available-tools)



## Overview

This repository includes various scripts related to camera calibration and image processing. Each tool is organized into its own directory with a detailed README file explaining its purpose and usage.

## Installation

Ensure you have Python 3.8+ installed along with the required dependencies. The necessary libraries can be installed using:

```bash
pip install numpy opencv-python pyyaml matplotlib scikit-image
```

Additional dependencies may be required depending on the specific tools used. Check the README file within each directory for more details.

## Available Tools

The repository currently contains the following tools:

- **Inverse Polynomial Computation for Cartesian Polynomial Model (CPM)**
  - Computes the inverse polynomial transformation for a camera using the CPM model.
  - Generates JSON files with inverse polynomial coefficients.
  - See the `inverse_polynomial_coefficients/README.md` for more details.

- **Image Undistortion and Comparison Tools**
  - Scripts for undistorting images using different camera models.
  - Tools for comparing images using Structural Similarity Index (SSIM).
  - See the `image_processing/README.md` for more details.

---

For further details on each tool, please refer to the individual README files in the respective directories.

