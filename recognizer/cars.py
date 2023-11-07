import concurrent.futures
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches

from recognizer import *

from pathlib import Path

__BASE_PATH__ = Path(__file__).resolve().parent


class Vehicle:
    def __init__(self):
        self.i_never_use_staticmethod = True
        # load model #
        self.model = None
        # Set marker setting #
        self.min_score_thresh = os.environ.get('MARK__MIN_SCORE_THRESH')
        self.min_score_thresh = float(self.min_score_thresh) if self.min_score_thresh is not None else 0.45
        # Set style #
        self.font_scale = 1
        self.font_color = (0, 0, 255)  # BGR
        self.font_alpha = 0.5
        self.font_width = 2
        self.box_color = (255, 0, 255)
        self.box_alpha = 1
        self.box_line = 1

        # - for matplotlib
        def __to_matplotlib_color(bgr):
            return tuple(channel / 255 if channel != 0 else 0 for channel in reversed(bgr))

        self.font_properties = {
            'size': self.font_scale,
            'weight': self.font_width,
            'color': __to_matplotlib_color(self.font_color)
        }
        self.rectangle_properties = {
            'linewidth': self.box_line,
            'edgecolor': __to_matplotlib_color(self.box_color),
            'facecolor': 'none',
            'alpha': self.box_alpha
        }

    def _convert_np_obj(self, datas):
        """
        Recursively convert numpy objects to Python native objects within a nested structure.
        It handles nested dictionaries, lists, and individual numpy objects.

        :param datas: The input object can be of any type.
        :return: The converted object with all numpy objects converted to Python native objects.
        """
        if isinstance(datas, dict):  # If the object is a dictionary
            return {key: self._convert_np_obj(value) for key, value in
                    datas.items()}  # Recursively convert each value in the dictionary
        elif isinstance(datas, list):  # If the object is a list
            return [self._convert_np_obj(element) for element in datas]  # Recursively convert each element in the list
        elif isinstance(datas, np.generic):  # If the object is a numpy generic type (includes numpy scalars)
            return datas.item()  # Convert numpy scalar to a Python native type
        elif isinstance(datas, np.ndarray):  # If the object is a numpy array
            return datas.tolist()  # Convert numpy array to a Python list
        else:  # If the object is any other type
            return datas  # Return the object as is

    def load_model(self, model_path):
        """
        Load a model.

        Models:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

        :param model_path:
        :return:
        """

        self.model = tf.saved_model.load(model_path)
        # print("loading model")

    def mark_vehicles(self, image, convert_np_data=False):
        """
        Return the vehicle's marking data.

        image: cv2.imread(img_path)

        :param convert_np_data:
        :param image:
        :return:
        """

        str_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Precondition images
        # print("preconditioning")
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

        # Tensor
        # print("convert to tensor")
        input_tensor = tf.convert_to_tensor([image])
        detections = self.model(input_tensor)

        # Find results
        # print("num detections")
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Mark cars
        response = {
            "data": []
        }
        for i in range(num_detections):
            score = detections['detection_scores'][i]
            if score >= self.min_score_thresh:
                class_id = detections['detection_classes'][i]

                # target vehicles class
                vehicle_ids = [3, 4, 8, 6]
                if class_id in vehicle_ids:
                    class_name = coco_class_ids[class_id]
                    detection = detections['detection_boxes'][i]

                    result = {
                        "class": {
                            "id": class_id,
                            "name": class_name
                        },
                        "detection": detection,
                        "box": {
                            "y": [
                                detection[0],
                                detection[2]
                            ],
                            "x": [
                                detection[1],
                                detection[3]
                            ]
                        },
                        "score": score,
                        "frame": np.append(detection, [score])
                    }
                    response["data"].append(result)

        response["usage"] = {
            "time": time.time() - str_time
        }

        return self._convert_np_obj(response) if convert_np_data else response

    def ___plot_detections(self, image, mark_vehicles):
        """
        Return a cars-marked plt object.

        image: cv2.imread(img_path)

        :param image:
        :param mark_vehicles:
        :return:
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # create matplotlib image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for detection in mark_vehicles:
            # Retrieve the normalized bounding box coordinates
            y_norm = detection['box']['y']
            x_norm = detection['box']['x']

            # Calculate the actual pixel coordinates
            xmin = x_norm[0] * image.shape[1]
            xmax = x_norm[1] * image.shape[1]
            ymin = y_norm[0] * image.shape[0]
            ymax = y_norm[1] * image.shape[0]

            # Calculate the width and height of the bounding box
            width = (xmax - xmin)
            height = (ymax - ymin)

            # Create a rectangle to represent the bounding box
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                **self.rectangle_properties
            )

            # Add the rectangle to the image
            ax.add_patch(rect)

            # Add the class name and confidence score next to the bounding box
            plt.text(
                xmin, ymin, f"{detection['class']['name']} {detection['score']:.3f}",
                **self.font_properties
            )

        return plt

    def cutout_car_pictures(self, image, mark_vehicles):
        """
        Cut out the car pictures.

        image: cv2.imread(img_path)

        :param image:
        :param mark_vehicles:
        :return:
        """

        self.i_never_use_staticmethod = True  # :D

        image = plt.imread(image)

        # create matplotlib image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        car_images = []
        for detection in mark_vehicles:
            # Retrieve the normalized bounding box coordinates
            y_norm = detection['box']['y']
            x_norm = detection['box']['x']

            # Calculate the actual pixel coordinates
            xmin = int(x_norm[0] * image.shape[1])
            xmax = int(x_norm[1] * image.shape[1])
            ymin = int(y_norm[0] * image.shape[0])
            ymax = int(y_norm[1] * image.shape[0])

            # Cut it.
            cut_image = image[ymin:ymax, xmin:xmax]

            car_images.append(cut_image)

        return car_images


class VehicleNumberPlate:
    def _generate_parameter_sets(self):
        parameter_sets = []
        epsilon_step = 0.01  # 你可以根据需要调整这个值
        canny_step = 10  # 你可以根据需要调整这个值

        epsilon_start, epsilon_end = self.epsilon_range
        canny_start, canny_end = self.canny_thresholds

        for epsilon in self._frange(epsilon_start, epsilon_end, epsilon_step):
            for lower_threshold in range(canny_start[0], canny_start[1] + 1, canny_step):
                for upper_threshold in range(canny_end[0], canny_end[1] + 1, canny_step):
                    parameter_sets.append({
                        'epsilon': epsilon,
                        'lower_threshold': lower_threshold,
                        'upper_threshold': upper_threshold
                    })

        return parameter_sets

    @staticmethod
    def _frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    def __init__(self, image_array, epsilon_range=(0.02, 0.05), canny_thresholds=((10, 100), (50, 200))):
        self.image_array = image_array
        self.epsilon_range = epsilon_range
        self.canny_thresholds = canny_thresholds
        self.parameter_sets = self._generate_parameter_sets()

        if image_array is None:
            raise ValueError("Image array cannot be None")

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        if len(image_array.shape) == 3:  # If colored image
            self.gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image_array

    def detect_plates(self, epsilon, canny_threshold):
        edges = cv2.Canny(self.gray, *canny_threshold)

        # 显示边缘
        plt.imshow(self.gray, cmap='gray')
        plt.title(f'Original Image: {str(canny_threshold)}')
        # plt.show()

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        plates = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            if len(approx) >= 4:
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                if 2 < aspect_ratio < 10:
                    pts_src = box.astype(np.float32)
                    pts_dst = np.array([[0, 0], [33, 0], [33, 16.5], [0, 16.5]], dtype=np.float32)
                    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                    warped = cv2.warpPerspective(self.image_array, matrix, (33, 16.5))
                    plates.append(warped)
        return plates

    def detect_plates_concurrent(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for epsilon in np.linspace(*self.epsilon_range, num=5):  # Adjust num as needed
                for canny_threshold in self.canny_thresholds:
                    futures.append(executor.submit(self.detect_plates, epsilon, canny_threshold))

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    plates = future.result()
                    results.extend(plates)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
        return results


def __test__():
    img_path = os.path.join(__BASE_PATH__, "../static/samples/high_cam.jpg")
    model = os.path.join(__BASE_PATH__, '../saved_models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8')
    model = os.path.join(model, "saved_model")

    m = Vehicle()
    m.load_model(model)

    # Load Images
    print("loading image")
    img = cv2.imread(img_path)

    # #### Cars ##### #
    str_time = time.time()

    data = m.mark_vehicles(img)
    print(data)
    print("Usage:", time.time() - str_time)

    # draw boxes
    # m.__plot_detections(img, data)

    # Cut out - Car photos
    car_images = m.cutout_car_pictures(img_path, data)
    for car_image in car_images:
        plt.imshow(car_image)
        plt.show()

    # #### Number Plates ##### #
    for car_image in car_images:
        try:
            number_plate_detector = VehicleNumberPlate(car_image)
            number_plates = number_plate_detector.detect_plates_concurrent()

            # 显示结果
            for plate in number_plates:
                plt.imshow(plate)
                plt.axis('off')
                plt.show()
        except ValueError as e:
            print(e)
            continue


if __name__ == '__main__':
    __test__()
