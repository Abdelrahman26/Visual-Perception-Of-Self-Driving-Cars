import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import DataFetching as DF
import DatasetClass as DSC

sample_image_PATH = os.path.join(DF.train_dir, DF.train_lst[0])
#PATH to PIL
sample_image = Image.open(sample_image_PATH)
#PIL to np
sample_image_np = np.array(sample_image)
cityscape, label = DSC.split_image(sample_image_np)

num_classes = 10
label_model = KMeans(n_clusters=num_classes)
label_model.fit(label.reshape(-1, 3))







