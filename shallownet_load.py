from pyimagesearch.datasets import SimpleDataLoader
from pyimagesearch.preprocessing import ImageToArrayPreprocess, SimplePreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="animals", help="path to dataset folder")
ap.add_argument("-m", "--model", type=str, default="output/animals_weight.hdf5", help="path to model file")
args = vars(ap.parse_args())

classLabels = ["cat", "dog", "panda"]

print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocess()

sdl = SimpleDataLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained model")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, f"Label: {classLabels[preds[i]]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

