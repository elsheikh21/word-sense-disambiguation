import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow_hub as hub
from tensorflow.keras.backend import set_session
from tensorflow.keras.layers import Layer


class ElmoEmbeddingLayer(Layer):
    '''
    Integrate ELMo Embeddings from tensorflow hub into a
    custom Keras layer, supporting weight update.
    references: 1. https://github.com/strongio/keras-elmo
                2. https://tfhub.dev/google/elmo/2

    Session issue was solved using:
    1. https://github.com/tensorflow/hub/blob/master/docs/common_issues.md#running-inference-on-a-pre-initialized-module
    '''

    def _init_(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self)._init_(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2',
                               trainable=self.trainable,
                               name=f"{self.name}_module")
        self.set_elmo_session()
        self._trainable_weights += tf.trainable_variables(
            scope=f"^{self.name}_module/.*")
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return self.elmo(B.squeeze(B.cast(x, tf.string), axis=1),
                         as_dict=True,
                         signature='default',
                         )['elmo']

    def compute_mask(self, inputs, mask=None):
        return B.not_equal(inputs, '<PAD>')

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.dimensions

    @staticmethod
    def set_elmo_session():
        config = tf.ConfigProto()
        # dynamically grow the memory used on the GPU
        config.gpu_options.allow_growth = True
        # to log device placement (on which device the operation ran)
        config.log_device_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.Session(config=config)
        # set this TensorFlow session as the default session for Keras
        set_session(sess)
        # Init all vars
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
