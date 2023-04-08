
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

# =============================================================================
#  Model path from model directory and load model
# =============================================================================

path = 'Path to model from model directory'


model=tf.keras.models.load_model(path)

# =============================================================================
# Layer name for gradCAM, layer name will be different for different models
# Please refer to readme document for appropriate layer name 
# =============================================================================

layer_name = "Layer name for gradCAM"

# =============================================================================
#  Load the image for gradCAM anamlysis
# =============================================================================

img_path='image path for gradCAM'

img_path_original=img_path

image = np.array(load_img(img_path, target_size=(224,224, 3)))
img_resize= np.array(load_img(img_path_original))
original_img = image
original_img=tf.keras.preprocessing.image.img_to_array(original_img)
img = np.expand_dims(original_img, axis=0)

# =============================================================================
#  use appropriate pre-procession function based on the model, for example here 
#  we use densenet model preprocessing function
# =============================================================================

img=tf.keras.applications.densenet.preprocess_input(img)
img_array=img    
prediction = model.predict(img)

prediction_idx = np.argmax(prediction)

target_class=prediction_idx

img_resize_x=tf.keras.preprocessing.image.img_to_array(img_resize)
img_resize_y=np.expand_dims(img_resize_x, axis=0)
img_resize_y=img_resize_y/255
    

eps=1e-8

# =============================================================================
#  gradCAM anamlysis
# =============================================================================

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()
   

gradModel = Model(
			inputs=[model.inputs],
			outputs=[model.get_layer(layer_name).output,
				model.output])
    
with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
      inputs = tf.cast(img, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, prediction_idx]
		# use automatic differentiation to compute the gradients
grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
castConvOutputs = tf.cast(convOutputs > 0, "float32")
castGrads = tf.cast(grads > 0, "float32")
guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
convOutputs = convOutputs[0]
guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
(w, h) = (img_resize_y.shape[2], img_resize_y.shape[1])
heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
numer = heatmap - np.min(heatmap)
denom = (heatmap.max() - heatmap.min()) + eps
heatmap = numer / denom
activation_map=heatmap
    # heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
activation_map = np.uint8(255 * activation_map)

    # Convert to Heatmap
heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
interpolant=0.65
assert (
        interpolant > 0 and interpolant < 1
    ), "Heatmap Interpolation Must Be Between 0 - 1"


plt.rcParams["figure.dpi"] = 300
plt.axis('off')
c=np.uint8(img_resize * interpolant + cvt_heatmap * (1 - interpolant))
c=cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
plt.imshow (np.uint8(img_resize * interpolant + cvt_heatmap * (1 - interpolant)))


plt.savefig('image save path',bbox_inches='tight',transparent=True, pad_inches=0)
