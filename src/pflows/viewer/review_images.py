import json
import sys
import cv2
import os
import numpy as np
import yaml
import os.path


def putTextWithShadow(
    image,
    label,
    position,
    font,
    font_scale,
    text_color,
    shadow_color,
    text_thickness=1,
    shadow_thickness=1,
    shadow_offset=2,
):
    shadow_position = (position[0] + shadow_offset, position[1] + shadow_offset)

    # Draw shadow
    cv2.putText(image, label, shadow_position, font, font_scale, shadow_color, shadow_thickness)

    # Draw text
    cv2.putText(image, label, position, font, font_scale, text_color, text_thickness)


def load_original_image(image_path):
    original_image_path = image_path.replace("/yolo/train/images", "/raw/train/images")
    original_image_path = original_image_path.replace("/yolo/test/images", "/raw/test/images")
    original_image_path = original_image_path.replace("/yolo/val/images", "/raw/val/images")

    if not os.path.exists(original_image_path):
        original_image_path = image_path
    return original_image_path


def create_legend(classes, colors):
    # add ALL category
    classes = ["ALL"] + classes
    legend_image = np.zeros((len(classes) * 25, 200, 3), dtype=np.uint8)
    for i, class_name in enumerate(classes):
        if i == 0:
            color = (255, 255, 255)
        else:
            color = colors[i - 1]
        cv2.rectangle(legend_image, (0, i * 25), (25, (i + 1) * 25), color, -1)
        cv2.putText(
            legend_image,
            class_name,
            (30, i * 25 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return legend_image


def load_colors(classes):
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy Blue
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (255, 192, 203),  # Pink
        (165, 42, 42),  # Brown
        (240, 230, 140),  # Khaki
        (173, 216, 230),  # Light Blue
        (240, 128, 128),  # Light Coral
        (144, 238, 144),  # Light Green
        (224, 255, 255),  # Light Cyan
    ]
    colors = colors * (len(classes) // len(colors) + 1)
    return colors[: len(classes)]


def load_classes(image_path):
    yaml_dir = image_path.split("/images/")[0]
    if yaml_dir == image_path:
        yaml_dir = os.path.dirname(image_path)

    classes = []
    try:

        yaml_file = open(f"{yaml_dir}/../data.yaml", "r")
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        classes = parsed_yaml_file["names"]
        if isinstance(classes, dict):
            classes = list(classes.values())
        return [str(cls) for cls in classes]
    except:
        try:
            classes_file = open(f"{yaml_dir}/../classes.txt", "r")
            return classes_file.read().splitlines()
        except:
            try:
                print(f"{yaml_dir}/../../classes.txt")
                classes_file = open(f"{yaml_dir}/../../classes.txt", "r")
                return classes_file.read().splitlines()
            except:
                return []


def get_label_path(image_path):
    label_path = image_path.replace(".jpg", ".txt").replace("/images/", "/labels/")
    confidence_path = image_path.replace(".jpg", ".txt").replace("/images/", "/confidences/")
    if os.path.exists(label_path):
        if confidence_path == image_path or confidence_path == label_path:
            confidence_path = None
        return label_path, confidence_path
    label_path = image_path.replace(".jpg", ".txt")
    if os.path.exists(label_path):
        return label_path, None
    raise Exception("No label path found")


def get_color(classes, colors, class_id):
    if isinstance(classes, dict):
        return colors[classes[class_id]]
    return colors[class_id]


def load_image(image_path, selected_category, legend_image, classes, colors, initial=False):
    if initial:
        print("image_path", image_path)
    original_image_path = load_original_image(image_path)

    label_path, confidence_path = get_label_path(image_path)
    info_path = image_path.replace(".jpg", ".json").replace("/images/", "/info/")
    if info_path and info_path != image_path and os.path.exists(info_path):
        info = json.loads(open(info_path, "r").read())
        print("\tsource:", info.get("path"))
    # Load the image
    image = cv2.imread(original_image_path)

    # YOLOv8 output format for a single bounding box: [class_id, center_x, center_y, width, height]
    # You'll need to replace this with your actual YOLOv8 output data

    # read yolo output open the file and split spaces in each line
    yolo_output = []
    category_ids = set()

    with open(label_path, "r") as file:
        for line in file:
            yolo_output.append([float(x) for x in line.split(" ")])
            category_ids.add(line.split(" ")[0])
    if not classes:
        classes = dict(
            [(int(category_id), index) for index, category_id in enumerate(list(category_ids))]
        )
        colors = load_colors(classes.values())

    yaml_dir = image_path.split("/images/")[0]
    if yaml_dir == image_path:
        yaml_dir = os.path.dirname(image_path)

    confidences = None
    if confidence_path and os.path.exists(confidence_path):
        confidences = [
            float(confidence or 0) for confidence in open(confidence_path, "r").read().splitlines()
        ]

    for index, box in enumerate(yolo_output):
        left = top = right = bottom = 0

        # Define the color for the bounding box based on class_id (you can customize this)
        # color = (0, 255, 0)  # Green
        # color = (255, 255, 0)  # Blue
        color = (153, 255, 255)  # Yellow
        # color = (0, 0,0) # Black

        class_id = int(box[0])
        if selected_category is not None and class_id != selected_category:
            continue
        if len(box) > 5:
            # We are going to draw the segmentation mask
            points = box[1:]
            points = np.array(points).reshape(-1, 2)
            points = points * np.array([image.shape[1], image.shape[0]])
            points = points.astype(int)
            cv2.polylines(
                image,
                [points],
                isClosed=True,
                color=get_color(classes, colors, class_id),
                thickness=2,
            )
            # get left, top, right, bottom
            left = np.min(points[:, 0])
            top = np.min(points[:, 1])
            right = np.max(points[:, 0])
            bottom = np.max(points[:, 1])

        else:
            _, center_x, center_y, width, height = box

            # Convert YOLO format to pixel coordinates
            img_height, img_width, _ = image.shape
            left = int((center_x - width / 2) * img_width)
            top = int((center_y - height / 2) * img_height)
            right = int((center_x + width / 2) * img_width)
            bottom = int((center_y + height / 2) * img_height)

            # Draw the bounding box on the image
            cv2.rectangle(image, (left, top), (right, bottom), get_color(classes, colors, class_id))

        # Optionally, you can add text to label the object with its class ID
        label = str(classes[int(class_id)])
        putTextWithShadow(
            image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, (0, 0, 0)
        )

        if confidences is not None:
            confidence = f"{confidences[index]:.2f}"
            putTextWithShadow(
                image, confidence, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, (0, 0, 0)
            )
    cv2.imshow("Legend", legend_image)
    cv2.imshow("YOLOv8 Output", image)


def main(target_path):
    # Check if the target path exists and if it's a folder
    if not os.path.exists(target_path):
        print(f"Invalid path: {target_path}")
        return
    if not os.path.isdir(target_path):
        images = [target_path]
    else:
        images = [
            os.path.join(target_path, image)
            for image in os.listdir(target_path)
            if image.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    classes = load_classes(target_path)
    colors = load_colors(classes)

    legend_image = create_legend(classes, colors)
    selected_category = None
    processed_image = None
    mouse_inside_legend = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_category, mouse_inside_legend
        nonlocal processed_image
        if event == cv2.EVENT_MOUSEMOVE:
            if x < legend_image.shape[1] and y < legend_image.shape[0]:
                mouse_inside_legend = True
                new_category = y // 25
                if new_category == 0:
                    selected_category = None
                else:
                    selected_category = new_category - 1
                processed_image = load_image(
                    images[current_image_index], selected_category, legend_image, classes, colors
                )

    legend_window_name = "Legend"
    output_window_name = "YOLOv8 Output"

    cv2.namedWindow(legend_window_name)
    cv2.namedWindow(output_window_name)

    legend_window_width = legend_image.shape[1]

    # Set the positions of the windows
    cv2.moveWindow(legend_window_name, 100, 100)
    cv2.moveWindow(output_window_name, 100 + legend_window_width + 10, 100)

    cv2.setMouseCallback("Legend", mouse_callback)

    current_image_index = 0
    while True:
        print(f"Image {current_image_index + 1}/{len(images)}")
        if processed_image is None:
            processed_image = load_image(
                images[current_image_index],
                selected_category,
                legend_image,
                classes,
                colors,
                initial=True,
            )
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == 3:  # Left arrow
            current_image_index = (current_image_index + 1) % len(images)
        elif key == 2:  # Right arrow
            current_image_index = (current_image_index - 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])
