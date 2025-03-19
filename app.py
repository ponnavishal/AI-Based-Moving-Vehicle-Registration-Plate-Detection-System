import os
import cv2
import numpy as np
import imutils
import easyocr
import csv
import streamlit as st
from PIL import Image

# Streamlit UI
st.title("Automatic License Plate Recognition (ALPR)")
st.write("Upload an image of a vehicle to detect its license plate and retrieve details.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Locate license plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        # Create mask and extract plate region
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Crop the detected plate
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Apply thresholding to improve OCR accuracy
        cropped_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR Read License Plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if result:
            text = result[0][-2].upper()  # Convert to uppercase
            st.subheader("Detected License Plate:")
            st.write(f"📸 License Plate Number: **{text}**")

            # Draw on the original image
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60),
                              fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

            # Display processed image
            st.image(res, caption="Detected License Plate", use_column_width=True)

            # Vehicle Data Lookup Function
            def get_vehicle_details(plate_number, filename):
                with open(filename, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['plate number'] == plate_number:
                            return {
                                "State Name": row["state name"],
                                "Owner Name": row["owner name"],
                                "Verification": row["verification"],
                                "Brand": row["brand"],
                                "Model": row["model"],
                                "Color": row["color"],
                                "Chassis No.": row["ch.no"]
                            }
                return None

            # Define CSV File Path (Replace with actual file path)
            filename = r"C:\Users\india\OneDrive - Chandigarh University\Desktop\ml projects\moving vehicle detection system\data (1).csv"

            # Get vehicle details
            vehicle_details = get_vehicle_details(text, filename)

            if vehicle_details:
                st.subheader("Vehicle Details:")
                st.write(f"🚘 **Plate Number:** {text}")
                st.write(f"🏙 **State Name:** {vehicle_details['State Name']}")
                st.write(f"👤 **Owner Name:** {vehicle_details['Owner Name']}")
                st.write(f"✅ **Verification:** {vehicle_details['Verification']}")
                st.write(f"🏎 **Brand:** {vehicle_details['Brand']}")
                st.write(f"🚗 **Model:** {vehicle_details['Model']}")
                st.write(f"🎨 **Color:** {vehicle_details['Color']}")
                st.write(f"🔍 **Chassis No.:** {vehicle_details['Chassis No.']}")
            else:
                st.warning("⚠ Plate number not found in the database.")

        else:
            st.error("❌ No text detected from the license plate.")
    else:
        st.error("❌ No license plate detected.")

