import argparse
import numpy as np
import logging
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_poly_inv(taylor_coefficient, nb_sample=100, sample_ratio=0.9, max_degree_inverse_poly=40):
    if taylor_coefficient is None:
        raise ValueError("Camera parameters are empty.")

    if not (0 <= sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be between 0 and 1. Current value: {sample_ratio}")

    logger.info("Starting inverse function estimation...")
    
    theta = np.linspace(0, np.pi * sample_ratio, nb_sample)
    rho = []
    
    for i in range(nb_sample):
        taylor_tmp = taylor_coefficient[::-1].copy()
        taylor_tmp[-2] -= np.tan(np.pi / 2 - theta[i])
        roots = np.roots(taylor_tmp)
        roots = roots[(roots > 0) & (np.imag(roots) == 0)]
        roots = np.array([float(np.real(e)) for e in roots])
        rho.append(np.min(roots) if roots.shape[0] > 0 else np.nan)

    rho = np.array(rho)
    mask = ~np.isnan(rho)
    theta_valid = theta[mask]
    rho_valid = rho[mask]

    max_error = float("inf")
    deg = 1
    
    while max_error > 0.01 and deg < max_degree_inverse_poly:
        inv_coefficient = np.polyfit(theta_valid, rho_valid, deg)
        rho_inv = np.polyval(inv_coefficient, theta_valid)
        max_error = np.max(np.abs(rho_valid - rho_inv))
        deg += 1

    logger.info(f"Reprojection error: {max_error:.4f}")
    logger.info(f"Inverse polynomial degree: {deg}")
    logger.info(f"Inverse coefficients: {inv_coefficient}")
    
    return inv_coefficient.tolist()

def load_parameters(file_path):
    """Load parameters from a file and return them as a dictionary."""
    params = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and ':' in line:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                # Convert the value to float or int based on its content
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    logger.warning(f"Skipping invalid value for {key}: {value}")
                    continue
                params[key] = value
    return params

def save_inverse_parameters(input_file, inv_coeff, params):
    """Save the inverse parameters to a new file in JSON format."""
    output_file = os.path.splitext(input_file)[0] + "_inverse.json"
    date_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    inverse_data = {
        "date": date_str,
        "camera_name": "panoramis",
        "valid": [True] * 3 + [False] * 10,  # Example validity array
        "taylor_coefficient": [params['a0'], params['a1'], params['a2'], params['a3'], params['a4']],
        "distortion_center": [params['u0'], params['v0']],
        "stretch_matrix": [[1, 0], [0, 1]],
        "inverse_poly": inv_coeff
    }
    
    with open(output_file, "w") as file:
        json.dump(inverse_data, file, indent=4)
    
    logger.info(f"Inverse parameters saved in {output_file}")
    

def main(args):
    # Load parameters
    params = load_parameters(args.parameters)
    
    # Prepare camera model data
    taylor_coefficient = np.array([params['a0'], params['a1'], params['a2'], params['a3'], params['a4']])

    # Calculate inverse parameters
    inv_coeff = find_poly_inv(taylor_coefficient)
    
    # Save results to a new file
    save_inverse_parameters(args.parameters, inv_coeff, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process camera parameters.")
    parser.add_argument('--parameters', '-p', type=str, required=True, help="Path to the parameter file.")
    args = parser.parse_args()
    
    main(args)


