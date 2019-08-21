# StyleNET
# Copyright (C) 2019 John Gouwar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# imageProcessing.py - Functions to prepare input images and write output images

import os
import cv2
import numpy as np
import tensorflow as tf

def process_inputs(style_path, content_path, content_resize_factor):
    '''
    Packs the input images into NHWC numpy arrays with values between 0.0 and
    1.0

    Parameters
      style_path: a valid path to the style image file
      content_path: a valid path to the content image file
      content_resize_factor: the factor by which the content image will be
                              rescaled by

    Returns
      images: a dict whose keys are 'style' and 'content' and whose values are
               the respective images as 1xHxWx3 float32 numpy arrays whose
               values are from 0.0 to 1.0. The images are in RGB order.

    Preconditions
      style_path and content_path point to images that can be read by cv2.imread
       (read opencv documentation for what sorts of image files these could be)

    Postconditions
      The style image is resized to be the same size as the content image after
      content resize factor has been applied
    '''
    # Read images and convert them to RGB
    content_image = cv2.imread(content_path)
    style_image = cv2.imread(style_path)

    # Resize the content image if necessary
    if content_resize_factor != 1:
        if content_resize_factor > 1:
            inter = cv2.INTER_CUBIC
        else:
            inter = cv2.INTER_AREA

        content_image = cv2.resize(content_image, (0,0), content_image,
                             fx=content_resize_factor, fy=content_resize_factor,
                             interpolation=inter)

    # Make the style image the same size as the content image
    if (style_image.shape != content_image.shape):
        shp = (content_image.shape[1], content_image.shape[0])
        style_image = cv2.resize(style_image, shp, style_image)


    # Pack the images into a dict in RGB order with NHWC format
    format_img = lambda img: np.float32(np.divide(img, 255.0))[:, :, ::-1]
    batch_img = lambda img: np.expand_dims(img, axis=0)
    images = {
                "style" : batch_img(format_img(style_image)),
                "content" : batch_img(format_img(content_image)),
             }


    return images


def write_outputs(final_output, checkpoint_outputs, output_dir, output_name,
                  output_extension):
    '''
    Writes the network outputs to images

    Parameters
      final_output: a 1xHxWx3 float32 tensor whose values are between 0.0 and
                     1.0 which represents the final synthesized image from the
                     network. The channels are in RGB order
      checkpoint_outputs: a dict whose keys are checkpoint numbers and whose
                           values are 1xHxWx3 float32 tensors in RGB order which
                           represent the state of the synthesized image at that
                           respective step
      output_dir: the directory where the output images will be written
      output_name: the name of final output image
      output_extension: the file extension of the final output image

    Returns
      None

    Preconditions
      output_dir is a directory which exists
      output_extension is a valid extension for cv2.imwrite (see opencv
       documentation for more details) and does not contain a '.'

    Postconditions
      A final image is written to output_dir/output_name.output_extension and
       every checkpoint image is written to
       output_dir/output_name_Check_CHECKNUM.output_extension.
      If files already exist with the above names, they are overwritten


    '''
    # Convert from NHWC float32 tensor in RGB to BGR uint8 image
    make_img = lambda img: np.uint8(np.multiply(img.numpy()[0, :, :, ::-1],255))

    for key in checkpoint_outputs:
        checkpoint_image = make_img(checkpoint_outputs[key])
        checkpoint_image_name = os.path.join(output_dir, output_name + "_Check_"
                                              + str(key) + "." +
                                         output_extension)
        cv2.imwrite(checkpoint_image_name, checkpoint_image)

    final_image = make_img(final_output)
    final_output_name = os.path.join(output_dir, output_name + "." +
                                     output_extension)
    cv2.imwrite(final_output_name, final_image)

    print("Outputs written to: %s" % output_dir)
