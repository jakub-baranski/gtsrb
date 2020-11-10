from skimage import transform, io, exposure

import numpy as np

from TSCNN import TSCNN

def get_test_image_paths():
    base_path = 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/'
    csv_path = f'{base_path}/GT-final_test.test.csv'

    with open(csv_path) as csv_file:
        rows = [r for r in csv_file.readlines()[1:]]

        rows = rows[:100]

        paths = []

        for row in rows:
            split = row.strip().split(';')
            image_path = split[0]
            paths.append(f'{base_path}{image_path}')

        return paths


def load_image(image_path: str):

        image = io.imread(image_path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)


image_paths = get_test_image_paths()

model = TSCNN().load_saved()
path_label_dict = {}

for path in image_paths[:10]:
    image = load_image(path)
    predictions = model.predict(image)
    highest_prediction = predictions.argmax(axis=1)[0]
    path_label_dict[path] = highest_prediction

print(path_label_dict)