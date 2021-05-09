import tensorflow as tf
import time
import numpy as np


class Model:
    """
    Life Model wrapper for efficient policy value rollouts
    """

    @classmethod
    def from_file(cls, path):
        return Model(tf.lite.Interpreter(model_path=path))

    @classmethod
    def from_keras(cls, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        lite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=lite_model)
        return Model(interpreter, model_content=lite_model)

    def save(self, path):
        """Serialize model"""
        if self.model_content:
            with open(f'{path}/model.tflite', 'wb') as f:
                f.write(self.model_content)
        else:
            print('Model already serialized')

    def __init__(self, interpreter, model_content=None):
        """ Set up model"""
        self.interpreter = interpreter
        self.model_content = model_content
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        # self.__cache__ = {}

    def predict(self, inp):
        """
        Predict single input, this is ok cuz we only ever use that
        """

        # currently with neural network architecture, caching does not
        # provide any significant advantage over recomputing due to
        # tflite being insanely fast, however if we change the architecture
        # of the neural network, it may be desirable to cache results.

        # tuple_inp = tuple(inp.flatten())
        # if tuple_inp in self.__cache__:
        #     print('cache')
        #     return np.array(tuple_inp, dtype=self.input_dtype).reshape(self.input_shape)

        inp = inp.reshape(self.input_shape)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)

        # self.__cache__[tuple_inp] = tuple(out.flatten())
        return out


if __name__ == '__main__':
    # Generate models
    start = time.time()
    test_model = Model.from_file('generated/initial/value/model.tflite')
    print(f'Model loaded in {time.time() - start}')
