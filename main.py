import cv2
import numpy as np
import pytesseract
import time
from picamera.array import PiRGBArray
from picamera import PiCamera


def filter_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result


def process_image(img):
    # Define the ROI (x, y, width, height) containing the decimal meter
    # roi = img[y:y+height, x:x+width]
    roi = img  # Modify this line if you want to crop the image

    red_filtered = filter_red(roi)

    gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def recognize_digits(img):
    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 -c load_system_dawg=false -c load_freq_dawg=false'
    return pytesseract.image_to_string(img, config=config)


def main():
    # Initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # Allow the camera to warm up
    time.sleep(0.1)

    while True:
        # Capture an image from the camera
        camera.capture(rawCapture, format="bgr")
        img = rawCapture.array

        processed_img = process_image(img)
        digits = recognize_digits(processed_img)
        print(f"Recognized digits: {digits}")

        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # Sleep for the desired output cycle (2 Hz or 4 Hz)
        time.sleep(0.5)  # Adjust this value for your desired output cycle

if __name__ == "__main__":
    main()
