import cv2
import os


def get_exposure(dataset_path, suffix, num_channels=3, debug=False):
    num_hists = 0
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if num_hists > 10 and debug:
                break
            if name.endswith(("JPEG")):
                image_file = os.path.join(root, name)
                img = cv2.imread(image_file, 0)
                hist = cv2.calcHist([img],[num_channels], None, [256], [0,256])
                print hist


if __name__ == "__main__":
    dataset_path = "/datasets/BigLearning/ahjiang/image-data/imagenet/"
    suffix = "JPEG"
    avg, var = get_exposure(dataset_path, suffix, True)
