from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

classifier = load_model('/home/pi/Documents/cnn/CNN_NUMBERS.h5')
test_image = image.load_img('/home/pi/Desktop/MNIST Dataset JPG format/MNIST - JPG - testing/2/236.jpg',
							target_size=(28,28),color_mode="grayscale")
							
test_image=image.img_to_array(test_image)
#expansion of dimesionality
test_image=np.expand_dims(test_image, axis=0)

result=classifier.predict(test_image)
index=(np.where(result==result.max())) #returns the index of the highest propability
prob=np.max(result) #returns the max propability
print(f"The given image bellongs to class {index} with a probability of {prob}")
