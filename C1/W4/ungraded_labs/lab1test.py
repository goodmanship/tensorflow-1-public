## NOTE: If you are using Safari and this cell throws an error,
## please skip this block and run the next one instead.
# I didn't try this.. i'd have to rewrite t for my local

import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

test_image_path = os.path.join('./horse-or-human/horses')

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = load_img(path, target_size=(300, 300))
  x = img_to_array(img)
  x /= 255
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
    
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
 