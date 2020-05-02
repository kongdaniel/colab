import tensorflow as tf

in_path = "./img/MobileFaceNet.pb"
out_path = "./pretrained_model/tflite/MobileFaceNet.tflite"

input_tensor_name = ["input"]
input_tensor_shape = {"input": [2, 112, 112, 3]}
classes_tensor_name = ["embeddings"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name,
classes_tensor_name, input_shapes=input_tensor_shape)
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)





# import tensorflow as tf
#
# convert = tf.lite.TFLiteConverter.from_frozen_graph("mfnnew.pb", input_arrays=["input"], output_arrays=["embeddings"],input_shapes={"input":[1,112,112,3]})
#
# convert.post_training_quantize = True
# tflite_model = convert.convert()
# open("model.tflite", "wb").write(tflite_model)

# #当需要给定输入数据形式时，给出输入格式：0
# import tensorflow as tf
# #tf.lite.TFLiteConverter  原函数
# path = "./"
# convert = tf.contrib.lite.toco_convert.from_frozen_graph(path + "mfn.pb", input_arrays=["images"],
#                                                     output_arrays=["output"],
#                                                     input_shapes={"images": [1, 540, 960, 1]})
# convert.post_training_quantize = True
# tflite_model = convert.convert()
# open(path + "quantized_model.tflite", "wb").write(tflite_model)
# print("finish!")