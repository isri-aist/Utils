import cv2
import numpy as np
import os
import argparse
from skimage.metrics import structural_similarity as ssim

def calculate_similarity(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    SSIM ranges from -1 to 1, where 1 means the images are identical.
    """
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    similarity_index, _ = ssim(image1_gray, image2_gray, full=True)
    return similarity_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--image1', required=True, help='Path to first input image')
    parser.add_argument('-i2', '--image2', required=True, help='Path to second input image')
    parser.add_argument('-o', '--output_dir', help='Directory to save the difference image')
    args = parser.parse_args()
    
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    if img1 is None or img2 is None:
        raise ValueError("One or both input images could not be loaded!")
    
    # Calculate SSIM and round to two decimal places
    ssim_value = calculate_similarity(img1, img2)
    #ssim_value = round(calculate_similarity(img1, img2), 4)
    print(f"SSIM: {ssim_value}")
    
    # Compute absolute difference
    diff_image = cv2.absdiff(img1, img2)
    
    # If no output_dir is provided, use the directory of image1
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.image1)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract image names without extensions
    image1_name = os.path.splitext(os.path.basename(args.image1))[0]
    image2_name = os.path.splitext(os.path.basename(args.image2))[0]
    
    # Generate output filename with SSIM value
    output_filename = f"diff_{image1_name}_{image2_name}_SSIM_{ssim_value}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the difference image
    cv2.imwrite(output_path, diff_image)
    print(f"Difference image saved at: {output_path}")
    
    # Display the difference image
    cv2.imshow("Difference Image", diff_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

