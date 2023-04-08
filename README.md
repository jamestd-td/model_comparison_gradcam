# model_comparison_gradcam
This work is to compare nine deep learning models for classifying TB vs Normal from Chest X-ray and generate GradCAM and their utility in decision making

Model Hyperparameters for training and Grad-CAM

Model        | N1 Layers     | N2 Layers     | Total Layers  | Layer Name for GradCAM  |
-------------| ------------- | ------------- | ------------- |  ---------------------- |
DenseNet121  |  117          |  289          |  430          | Conv5_block16_concat    |
DenseNet169  |  229          |  457          |  598          | Conv5_block32_concat    |
EfficientNetB3   | 123          |  196          |  389          | top_activation    |
EfficientNetB4  |  155          |  243          |  479          | top_activation    |
MobileNet V3   |  144         |  189          |  272          | Conv_1    |
NASNet Mobile  |  285          |  375          |  773          | Activation_187    |
ResNet 50   | 35          |  97          |  178          | Conv5_block3_out    |
ResNet101 V2  |  39         |  294         |  380          | Conv5_block3_out    |
VGG16  |  8          |  12          |  22          | Block5_pool    |
