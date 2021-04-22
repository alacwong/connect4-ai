import tensorflow as tf
import numpy as np


class Model:
    """
    Lite model for efficient policy value rollouts
    """

    @classmethod
    def from_file(cls, path):
        return Model(tf.lite.Interpreter(model_path=path))

    @classmethod
    def from_keras(cls, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        lite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=lite_model)
        return Model(interpreter)

    def __init__(self, interpreter):
        """ Set up model"""
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        """
        Predict single input, this is ok cuz we only ever use that
        """
        inp = np.array(inp, dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0][0]
