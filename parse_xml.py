import xml.etree.ElementTree as ET
import os
from sys import argv

def convert_cvat_xml_to_yolo(xml_file, output_dir="labels"):
    """
    Parses a CVAT XML file (containing masks or boxes) and converts annotations 
    to YOLO object detection format (txt files).
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 1. Extract Label Map
    # Map label names to class IDs (e.g., {"Pikachu": 0})
    labels_map = {}
    label_id = 0
    
    # CVAT usually lists labels under meta/job/labels
    for label_tag in root.findall(".//label"):
        name = label_tag.find("name").text
        if name not in labels_map:
            labels_map[name] = label_id
            label_id += 1
            
    print(f"Found classes: {labels_map}")

    # 2. Process Each Image
    count = 0
    for image in root.findall('image'):
        image_name = image.get('name')
        image_width = float(image.get('width'))
        image_height = float(image.get('height'))
        image_id = image.get('id')
        
        # The filename for the label usually corresponds to the image name
        # e.g., frame_000000.jpg -> frame_000000.txt
        file_base_name = os.path.splitext(image_name)[0]
        txt_filename = os.path.join(output_dir, f"{file_base_name}_{os.path.basename(xml_file).rsplit('.', 1)[0]}.txt")
        
        yolo_lines = []

        # Find all annotations for this image
        # In your file, they appear as <mask> tags, but we can treat them like boxes
        # because they have 'left', 'top', 'width', 'height' attributes.
        # We also check for standard <box> tags just in case.
        annotations = image.findall('mask') + image.findall('box')

        for ann in annotations:
            label_name = ann.get('label')
            if label_name not in labels_map:
                continue # Skip unknown labels

            class_id = labels_map[label_name]

            # Extract Pixel Coordinates
            # Some CVAT versions use xtl, ytl, xbr, ybr for boxes
            # Your masks use left, top, width, height
            if 'left' in ann.attrib:
                x_min = float(ann.get('left'))
                y_min = float(ann.get('top'))
                box_w = float(ann.get('width'))
                box_h = float(ann.get('height'))
            else:
                # Fallback for standard 'box' tag usually found in other CVAT exports
                xtl = float(ann.get('xtl'))
                ytl = float(ann.get('ytl'))
                xbr = float(ann.get('xbr'))
                ybr = float(ann.get('ybr'))
                x_min = xtl
                y_min = ytl
                box_w = xbr - xtl
                box_h = ybr - ytl

            # Calculate Center Coordinates
            x_center = x_min + (box_w / 2.0)
            y_center = y_min + (box_h / 2.0)

            # Normalize Coordinates (0 to 1)
            x_norm = x_center / image_width
            y_norm = y_center / image_height
            w_norm = box_w / image_width
            h_norm = box_h / image_height

            # Clamp values to ensure they are within [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            # YOLO Format: class_id x_center y_center width height
            yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Write to txt file if there are annotations
        if yolo_lines:
            with open(txt_filename, 'w') as f:
                f.write("\n".join(yolo_lines))
            count += 1

    print(f"Successfully converted {count} files to '{output_dir}/'.")
    
    # Create a classes.txt for reference
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        # Sort by ID to ensure correct order
        sorted_labels = sorted(labels_map.items(), key=lambda item: item[1])
        for name, _ in sorted_labels:
            f.write(f"{name}\n")

if __name__ == "__main__":
    xml_file_path = argv[1]
    convert_cvat_xml_to_yolo(xml_file_path, output_dir="dataset/labels")