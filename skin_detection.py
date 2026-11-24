
import cv2
import numpy as np

def detect_skin(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return "INVALID_IMAGE"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    total_pixels = mask.size
    skin_pixels = cv2.countNonZero(mask)

    percentage = (skin_pixels / total_pixels) * 100

    print(f"Skin-colored area: {percentage:.2f}%")

    if percentage >= 10:
        return "SKIN"
    else:
        return "NOT_SKIN"


if __name__ == "__main__":
   
    image_path = input("Enter image path: ").strip().strip('"')

    result = detect_skin(image_path)
    print("Detection Result:", result)
