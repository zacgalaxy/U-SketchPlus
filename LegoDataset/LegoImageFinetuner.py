""" 
Need to format all selected images to fir in a 256x256 boundry, To simplify the process I will select one of the grouped folders
And proceed to crop all images to 256x256. I will then compare them to the orginal photos to see if the cropping was done correctly.
And sleect those that are least distorted. I will still be selecting images based on. vartiaion and images quality. 

https://www.geeksforgeeks.org/image-resizing-using-opencv-python/

"""
import cv2
import matplotlib.pyplot as plt
import os 

inputroot="images_4_(Flowers)"
outputroot="Image_4_256x256"

new_w=256
new_h=256

os.makedirs(outputroot, exist_ok=True)
for root, dirs, files in os.walk(inputroot):
    # Compute relative path to maintain folder structure
    relative_path = os.path.relpath(root, inputroot)
    output_dir = os.path.join(outputroot, relative_path)
    # Ensure the corresponding folder exists in the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image file
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            # Load and resize image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping invalid image: {input_path}")
                continue

            resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_AREA)

            # Save resized image
            cv2.imwrite(output_path, resized_image)

print("Resizing complete!")


