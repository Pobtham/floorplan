import cv2
import os
import argparse
from ultralytics import YOLO
try:
    from google.colab.patches import cv2_imshow
    use_colab = True
except ImportError:
    use_colab = False

def display(image_path, model, names, color_map):
    """
    Reads an image from the given path, runs inference using the provided model,
    draws bounding boxes and labels on the image, and displays it.
    """
    # If image_path is a directory, choose the first valid image file.
    if os.path.isdir(image_path):
        files = os.listdir(image_path)
        # Filter for common image file extensions
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("No image files found in directory:", image_path)
            return
        image_file = os.path.join(image_path, image_files[0])
    else:
        image_file = image_path

    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print("Failed to read image:", image_file)
        return

    # Run inference on the image using the file path
    results = model(image_file)
    
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get the confidence score
            conf = box.conf[0] if hasattr(box, 'conf') else 0.0

            if conf < 0.5:
                continue

            if hasattr(box, 'cls'):
                class_id = int(box.cls[0])
                # If names is a dict use get(), else treat it as a list
                if isinstance(names, dict):
                    class_name = names.get(class_id, "N/A")
                else:
                    class_name = names[class_id] if class_id < len(names) else "N/A"
                label_text = f"{class_name}: {conf:.2f}"
                color = color_map.get(class_id, (255, 0, 0))
            else:
                label_text = f"{conf:.2f}"
                color = (255, 0, 0)  # Default color

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if use_colab:
        cv2_imshow(image)
    else:
        cv2.imshow("Detected Boundaries", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

model = YOLO("best.pt")
names = model.names
color_map = {0: (255, 51, 51), 1: (128, 255, 0), 2: (255, 0, 255), 3: (0, 102, 204)}
#display(f'{folder_path}/{file_list[1]}',model, names, color_map)
image_path = ''
display(image_path, model, names, color_map)
