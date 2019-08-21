# StyleNET
A Tensorflow implementation of Gatys et al. "Image Style Transfer with Convolutional Neural Networks" from CVPR 2016 (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

## Background
StyleNET takes two images, a style image and a content image, and renders the content image in the style of the style image. It does so by extracting the representations of each image from the intermediate outputs of the VGG19 network, pretrained on the ImageNet dataset.

Often the style image is a painting; however Gatys et al. do note that their algorithm is capable of photo-realistic style transfer as well. Incredibly textured paintings, like Impressionist pieces, seem to work best, given that their style is very distinct and does not rely as heavily on intricate details. The content image can be of anything, but do note that intricate details often get distorted by the style.

For more details on the intricacies of the algorithm, see the original paper Gatys et al.

## Requirements
This project requires the use of Python 3.x and TensorFlow 1.14. I use a TensorFlow version compiled from the binaries, thus I cannot guarantee that the version from PyPi will work.

Other required packages are OpenCV and Numpy, (see `requirements.txt` for exact versions).

## Running StyleNET
Running StyleNET is as simple as running the following command: `python styleNet.py --FLAG1=f1 --FLAG2=f2`

If you are using TensorFlow GPU, it can be helpful to set the environment variable `OPENCV_OPENCL_DEVICE=disabled` to keep opencv from hogging the GPU.

### Command Line Flags

#### style_image_dir
A path to the directory where style images are located. **Default: "../data/styleImages"**

#### content_image_dir
A path to the directory where content images are located. **Default: "../data/contentImages"**

#### style_image_name
The name of the style image file. **Default: "style_image.jpg"**

#### content_image_name
The name of the content image file. **Default: "content_image.jpg"**

#### num_steps
The number of transfer steps the program takes before writing the final image. **Default: 1000**

#### checkpoint_steps
The number of steps between image checkpoints. **Default: 100**

#### style_weight
The weighting factor of the contribution of the style loss to the total loss. **Default: 1e-2**

#### content_weight
The weighting factor of the contribution of the content loss to the total loss. **Default: 1e4**

#### style_layers
A comma separated string (with no spaces) of the layers used for style representations. **Default: "block1_conv1,block2_conv1,block3_conv1,block4_conv1,block5_conv1"**

#### content_layers
A comma separated string (with no spaces) of the layers used for content representations. **Default: "block4_conv2"**

#### content_resize_factor
The factor by which the content image is scaled. **Default: 1.0**

#### output_dir
A path to the directory where the output images are written. **Default: ""../outputs"**

#### output_name
The base name of the output images. **Default: "output"**

#### output_extension
The file extension for the output images (with no "."). **Default: png**

#### print_final_shape
Boolean flag whether to print final output shape before transfer steps are made. **Default: False**

## Acknowledgments
All credit for the algorithm goes to Gaty's et al.
