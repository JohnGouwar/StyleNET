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
# styleNet.py - Runs the style tranfer algorithm from the given command line
# flags

import tensorflow as tf
import cv2
import numpy as np
import os
import sys

from RepresentationExtractor import RepresentationExtractor
import imageProcessing

# Essentially makes this code function as close to TF2.xx as possible
tf.compat.v1.enable_resource_variables()
tf.compat.v1.enable_eager_execution()

###############################################################################
tf.app.flags.DEFINE_string("style_image_dir", "../data/styleImages",
                           """Directory where style images are located""")
tf.app.flags.DEFINE_string("content_image_dir", "../data/contentImages",
                            """Directory where content images are located""")
tf.app.flags.DEFINE_string("style_image_name", "style_image.jpg",
                            """Name of the style image file""")
tf.app.flags.DEFINE_string("content_image_name", "content_image.jpg",
                            """Name of the content image file""")
tf.app.flags.DEFINE_integer("num_steps", 1000,
                        """Number of transfer steps taken""")
tf.app.flags.DEFINE_integer("checkpoint_steps", 100,
                            """Number of steps between image checkpoints""")
tf.app.flags.DEFINE_float("style_weight", 1e-2,
                            """Weight of style loss""")
tf.app.flags.DEFINE_float("content_weight", 1e4,
                            """Weight of content loss""")
tf.app.flags.DEFINE_string("style_layers",
             "block1_conv1,block2_conv1,block3_conv1,block4_conv1,block5_conv1",
                            """Comma separated string of style layer names""")
tf.app.flags.DEFINE_string("content_layers", "block4_conv2",
                            """Comma separated string of content layer names""")
tf.app.flags.DEFINE_float("content_resize_factor", 1.0,
                          """Ratio content image is scaled by""")
tf.app.flags.DEFINE_string("output_dir", "../outputs",
                            """Directory where outputs are written.""")
tf.app.flags.DEFINE_string("output_name", "output",
                            """Base name of output images""" )
tf.app.flags.DEFINE_string("output_extension", "png",
                            """File extension of output images""")
tf.app.flags.DEFINE_boolean("print_final_shape", False,
                            """Flag to print final output shape""")
FLAGS = tf.app.flags.FLAGS
###############################################################################

def get_targets(extractor, style_image, content_image):
    '''
    Gets the content and style representations of the target images

    Parameters
      extractor: An instance of the RepresentationExtractor class
      style_image: An NHWC format image used for representing the final image's
                    style
      content_image: An NHWC format image used for representing the final
                      image's content

    Returns
      targets: a dict with the keys 'style' and 'content' where the value at
                each key is a dict whose keys are the layers of the CNN and
                values are the representations at those layers for style and
                content respectively

    Preconditions
      style_image and content_image tf.float32 tensors with all values between
       0 and 1

    Postconditions
      None (besides return)

    '''
    # Get the targets
    style_targets, dummy = extractor(style_image)
    dummy, content_targets = extractor(content_image)

    # Pack them into a dict
    targets = {'style': style_targets,
               'content': content_targets}

    return targets


def get_loss(output_style, output_content, style_targets, content_targets,
            style_weight, content_weight):
    '''
    Gets the total loss between the synthesized image and the targets

    Parameters
      output_style: a dict whose keys are layer names and whose values are the
                     style representations of the synthesized image at those
                     layers
      output_content: a dict whose keys are layer names and whose values are the
                       content representations of the synthesized image at those
                       layers
      style_targets: a dict whose keys are layer names and whose values are the
                       style representations of the style image at those layers
      content_targets: a dict whose keys are layer names and whose values are
                        the content representations of the content image at
                        those layers
      style_weight: a weighting factor of the contribution of the style loss to
                     the total loss
      content_weight: a weighting factor of the contribution of the content loss
                       to the total loss

    Returns
      total_ loss: a tf.float32 scalar tensor representing the total loss
                    weighted between the style and content loss

    Preconditions
      output_style must have the same keys as style_targets
      output_content must have the same keys as content_targets

    Postconditions
      None (besides return)
    '''
    # Style loss
    style_layer_weight = 1/len(output_style)
    style_loss = tf.add_n([tf.reduce_mean(tf.square(output_style[key] -
                                                    style_targets[key]))
                           for key in output_style])
    style_loss *= style_layer_weight


    # Content loss
    content_layer_weight = 1/len(output_content)
    content_loss = tf.add_n([tf.reduce_mean(tf.square(output_content[key] -
                                                      content_targets[key]))
                             for key in output_content])
    content_loss *= content_layer_weight

    # Final loss calculation
    total_loss = (style_weight * style_loss) + (content_weight * content_loss)
    return total_loss


@tf.function
def transfer_step(image, extractor, optimizer, targets, style_weight,
                    content_weight):
    '''
    A TensorFlow function to run a single step of the style transfer algorithm

    Parameters
      image: the current state of the synthesized image variable to be updated
              by this transfer step in NHWC format
      extractor: an instance of the RepresentationExtractor class
      optimizer: an instance of a TensorFlow optimizer
      targets: a dict with the keys 'style' and 'content' where the value at
                each key is a dict whose keys are the layers of the CNN and
                values are the representations at those layers for style and
                content respectively
      style_weight: a weighting factor of the contribution of the style loss to
                     the total loss
      content_weight: a weighting factor of the contribution of the content loss
                       to the total loss

    Returns
      None

    Preconditions
      image is a trainiable tf.float32 variable whose values are between 0 and 1
      extractor is the same instance of the extractor class used to create the
       targets

    Postconditions
      The image variable is updated in the direction of the gradient with
       respect to the loss
      The loss calculated with respect to the image is printed as a side effect
    '''

    # Extract the targets
    style_targets = targets['style']
    content_targets = targets['content']

    # Get the gradient with respect to loss
    with tf.GradientTape() as tape:
        output_style, output_content = extractor(image)
        loss = get_loss(output_style, output_content, style_targets,
                        content_targets,style_weight, content_weight)

    # Make a step in the direction of the gradient
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])

    # Make sure the image stays between 0 and 1
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

    tf.print("Loss:", loss, output_stream=sys.stdout)


def style_transfer(images, style_layers, content_layers, style_weight,
                    content_weight, num_steps, checkpoint_steps):
    '''
    Runs the style transfer algoritghm for the number of steps determined by the
    command line flags

    Parameters
      images: a dict with the keys 'style' and 'content' whose values are NHWC
               tf.float32 images with values between 0 and 1 of the style and
               content images respectively
      style_layers: a list of strings containing layer names to be used as the
                     extractor's style layers
      content_layers: a list of strings containing layer names to be used as the
                     extractor's content layers
      style_weight: a weighting factor of the contribution of the style loss to
                     the total loss
      content_weight: a weighting factor of the contribution of the content loss
                       to the total loss
      num_steps: the number of total transfer steps to be run
      checkpoint_steps: the number of steps between checkpoints

    Returns
      output_image: a tf.float32 tensor in NHWC format with values between 0 and
                     1 representing the final image synthesized by the style
                     transfer algorithm
      checkpoint_images: a dict whose keys are step numbers and the values are
                          tf.float32 tensors in NHWC format with values between
                          0 and 1 representing the state of synthesized image at
                          that particular step

    Preconditions
      No additional

    Postconditions
      None (besides returns)
    '''
    # Create the representation extractor from the appropriate layers
    extractor = RepresentationExtractor(style_layers, content_layers)

    # Get the target representations
    targets = get_targets(extractor, images["style"], images["content"])

    # Create the optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=.02, beta1=.99,
                                                  epsilon=1e-1)

    # Initialize the output image as the content image initialize the place to
    # store the checkpoints
    output_image = tf.Variable(images["content"], dtype=tf.float32)
    checkpoint_outputs = {}

    # Run the transfer algorithm
    for i in range(1, num_steps+1):
        print("Step %d:" % i)

        # Add to the checkpoints
        if i % checkpoint_steps == 0:
            checkpoint_outputs[str(i)] = tf.identity(output_image)

        # Run a transfer step
        transfer_step(output_image, extractor, optimizer, targets, style_weight,
                      content_weight)


    return output_image, checkpoint_outputs


def main(argv=None):
    '''
    Processes the images, runs the style transfer algorithm on them, and writes
    the outputs, all using the specified command line flags
    '''
    full_style_path = os.path.join(FLAGS.style_image_dir,
                                   FLAGS.style_image_name)
    full_content_path = os.path.join(FLAGS.content_image_dir,
                                     FLAGS.content_image_name)

    # Get images into NHWC, 0.0-1.0, format
    images = imageProcessing.process_inputs(full_style_path, full_content_path,
                                            FLAGS.content_resize_factor)

    # Print final shape if necessary
    if FLAGS.print_final_shape:
        print("Final image shape: {}".format(images['content'].shape))

    # Get a list of layer names
    style_layers = FLAGS.style_layers.split(",")
    content_layers = FLAGS.content_layers.split(",")

    # Run the style transfer algorithm
    final_output, checkpoint_outputs = style_transfer(images,
                                                      style_layers,
                                                      content_layers,
                                                      FLAGS.style_weight,
                                                      FLAGS.content_weight,
                                                      FLAGS.num_steps,
                                                      FLAGS.checkpoint_steps)

    # Write the output images
    imageProcessing.write_outputs(final_output, checkpoint_outputs,
                                  FLAGS.output_dir, FLAGS.output_name,
                                  FLAGS.output_extension)



if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)
