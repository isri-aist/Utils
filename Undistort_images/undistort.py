import cv2
import numpy as np
import argparse
import os
import yaml
import matplotlib.pyplot as plt


def load_camera_parameters(param_file):
    with open(param_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

def compute_K_new(params, factor):
    K = np.array([[params['fx'], 0, params['cx']], [0, params['fy'], params['cy']], [0, 0, 1]])
    return np.array([[K[0,0] * factor, 0, K[0,2]], [0, K[1,1] * factor, K[1,2]], [0, 0, 1]])

def undistort_image(image, params, factor=1.0, K_new=None):
    """
    Undistorts an image based on the camera model and distortion parameters.
    
    Parameters:
        image (numpy.ndarray): The input distorted image.
        params (dict): Dictionary containing camera parameters.
        factor (float): Scaling factor for the new camera matrix.
        K_new (numpy.ndarray, optional): New camera matrix. If None, it is computed based on the factor.
    
    Returns:
        numpy.ndarray: The undistorted image.
    """
    model = params['model']
    cols, rows = params['cols'], params['rows']
    print("Start undistortion with K_new =", K_new)

    if model == "OFM":
        K = np.array([[params['fx'], 0, params['cx']], [0, params['fy'], params['cy']], [0, 0, 1]])
        D = np.array([params['k1'], params['k2'], params['k3'], params['k4']])
        if K_new is None:
            K_new = np.array([[K[0,0] * factor, 0, K[0,2]], [0, K[1,1] * factor, K[1,2]], [0, 0, 1]])
        print("Start OFM undistortion with K_new =", K_new)
        undistorted_image = cv2.fisheye.undistortImage(image, K, D, None, K_new)
    
    elif model == "UCM":
        K = np.array([[params['fx'], 0, params['cx']], [0, params['fy'], params['cy']], [0, 0, 1]])
        if K_new is None:
            K_new = np.array([[K[0,0] * factor, 0, K[0,2]], [0, K[1,1] * factor, K[1,2]], [0, 0, 1]])
        D = np.zeros((4, 1))
        xi = np.array([params['xi']])
        print("Start UCM undistortion with K_new =", K_new)
        undistorted_image = cv2.omnidir.undistortImage(image, K, D, xi, Knew=K_new, flags=cv2.omnidir.RECTIFY_PERSPECTIVE)
    
    elif model == "EFM":
        h, w = rows, cols
        f, k, fov = float(params['f']), float(params['k']), float(params['fov'])
        u0, v0 = w // 2, h // 2
        theta_max = np.deg2rad(fov / 2)
        new_img = np.zeros_like(image)
        fx = 681 * factor 
        for v in range(h):
            for u in range(w):
                x, y = (u - 506) / fx, (v - 490) / fx
                theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), 1)
                if theta <= theta_max:
                    src_x = int(f / k * theta * x / np.sqrt(x**2 + y**2) + 503) if theta != 0 else u0
                    src_y = int(f / k * theta * y / np.sqrt(x**2 + y**2) + 492) if theta != 0 else v0
                    if 0 <= src_x < w and 0 <= src_y < h:
                        new_img[v, u] = image[src_y, src_x]
        print("Start EFM undistortion with K_new =", K_new)
        undistorted_image = new_img
    
    elif model == "RPM":
        K = np.array([[params['fx'], 0, params['cx']], [0, params['fy'], params['cy']], [0, 0, 1]])
        D = np.array([params['k1'], params['k2'], params['p1'], params['p2'], params['k3'], params['k4'], params['k5'], params['k6']])
        if K_new is None:
            K_new = np.array([[K[0, 0] * factor, 0, K[0, 2]], [0, K[1, 1] * factor, K[1, 2]], [0, 0, 1]])
        R = np.eye(3)
        P = np.array([[params['fx'] * factor, 0, params['cx'], 0], [0, params['fy'] * factor, params['cy'], 0], [0, 0, 1, 0]])
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, (cols, rows), cv2.CV_32FC1)
        print("Start RPM undistortion with K_new =", K_new)
        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    
    elif model == "CPM":
        raise ValueError(f"The camera model '{model}' is not supported. Please use inverse polynomial coefficient and py-OcamCalib")
    else:
        raise ValueError(f"The camera model '{model}' is not supported. Please check the parameters.")
    
    return undistorted_image

def process_images_in_directory(image_path, param_dir, factor, show=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(image_name)
    os.makedirs(output_dir, exist_ok=True)
    
    first_param_file = next((f for f in os.listdir(param_dir) if f.endswith('.txt')), None)
    if not first_param_file:
        raise ValueError(f"No calibration files found in {param_dir}")
    
    first_param_path = os.path.join(param_dir, first_param_file)
    first_params = load_camera_parameters(first_param_path)
    K_new = compute_K_new(first_params, factor)
    
    results = {"Original": image}
    
    for param_file in os.listdir(param_dir):
        if param_file.endswith('.txt'):
            param_path = os.path.join(param_dir, param_file)
            print(f"Processing: {param_path}")
            
            params = load_camera_parameters(param_path)
            undistorted_image = undistort_image(image, params, factor, K_new=K_new)
            
            output_path = os.path.join(output_dir, f"undistort_{os.path.splitext(param_file)[0]}_factor_{factor}.png")
            cv2.imwrite(output_path, undistorted_image)
            
            results[param_file] = undistorted_image
    
    if show:
        plot_images(results)

def plot_images(images):
    plt.figure(figsize=(10, 5))
    n = len(images)
    for i, (title, img) in enumerate(images.items()):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i + 1)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='Input image path')
    parser.add_argument('-p', '--parameters', required=True, help='Directory containing camera parameters files (.txt)')
    parser.add_argument('-f', '--factor', type=float, default=1.0, help='Scaling factor for K_new')
    parser.add_argument('-s', '--show', action='store_true', help='Show undistorted images')
    args = parser.parse_args()
    
    if not os.path.isdir(args.parameters):
        raise ValueError(f"The directory {args.parameters} does not exist.")
    
    process_images_in_directory(args.image, args.parameters, args.factor, args.show)

if __name__ == "__main__":
    main()

