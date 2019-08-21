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
# RepresentationExtractor - Class file describing the class which builds the
# model that has the appropriate outputs for the style and content layers

import tensorflow as tf

### HELPER FUNCTIONS ###
def _model_from_layers(layer_names):
    '''
    Build a subset of the VGG19 model from a list of layer names

    Parameters
      layer_names: a list of layer names for VGG19 network

    Returns
      model: a keras model whose input layers are the VGG19 input layers and
              output layers are the layers in layer names

    Preconditions
      All of the layers in layer_names must be valid names of layers in the
       Keras Applications implementation of the VGG19 network

    Postconditions
      None (besides return)
    '''
    base_model = tf.keras.applications.VGG19(include_top=False,
                                            weights="imagenet")
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=[base_model.input], outputs=outputs)
    model.trainiable = False
    return model


def _get_gram_matrix(layer_outputs):
    '''
    Takes a network output and transforms it into a Gram matrix

    Parameters
      layer_outputs: a NHWC tensor of the outputs from a layer of a neural
                      network

    Returns
      gram_matrix: the gram matrix for the given layer_outputs

    Preconditions
      None

    Postconditions
      None (besides return)
    '''
    shape = layer_outputs.shape.as_list()
    mat = tf.reshape(layer_outputs, (shape[-1], -1))
    gram_matrix = tf.linalg.matmul(mat, tf.transpose(mat))
    return gram_matrix


###############################################################################
class RepresentationExtractor(tf.keras.Model):
    '''
    A Keras Model subclass which extracts content and style representations from
    various layers of the VGG19 network from Keras Applications pretrained on
    the ImageNet dataset

    Instance Variables
      model: the special VGG19 model with output layers style_layers and
              content_layers
      style_layers: a list of layer names for the style representations
      content_layers: a list of layer names for the content representations
      num_style_layers: the number of style layers
      num_content_layers: the number of content layers
    '''

    def __init__(self, style_layers, content_layers):
        '''
        Constructor for RepresentationExtractor

        Parameters
          style_layers: a list of strings containing layer names to be used as
                         the extractor's style layers

          content_layers: a list of strings containing layer names to be used as
                           the extractor's content layers

        Preconditions
          All of the layers in style_layers and content_layers must be valid
           names of layers in the Keras Applications implementation of the VGG19
           network

        Postconditions
          An instance of the RepresentationExtractor class is created
        '''
        super(RepresentationExtractor, self).__init__(name=
                                                      "RepresentationExtractor")
        self.model = _model_from_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.model.trainiable = False


    def call(self, image):
        '''
        Calls the RepresentationExtractor network on an image

        Parameters
          image: a tf.float32 1xHxWx3 tensor with values between 0.0 and 1.0

        Returns
          style_representations: a dict whose keys are the extractor's
                                  style_layers and values are the Gram matrices
                                  of the outputs from those layers
          content_representations: a dict whose keys are the extractor's
                                    content_layers and values are the outputs
                                    from those layers

        Preconditions
          No additional

        Postconditions
          None (besides return)
        '''
        # Get the representations
        image = image * 255
        proc_image = tf.keras.applications.vgg19.preprocess_input(image)
        outputs = self.model(proc_image)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Build the output dicts
        style_representations = {}
        content_representations = {}
        for i, layer in enumerate(self.style_layers):
            style_representations[layer] = _get_gram_matrix(style_outputs[i])
        for i, layer in enumerate(self.content_layers):
            content_representations[layer] = content_outputs[i]

        # Return representations
        return style_representations, content_representations
