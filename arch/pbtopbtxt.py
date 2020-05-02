import tensorflow as tf
from tensorflow.python.platform import gfile


# 函数功能能,将pb模型转换为pbtxt,转换好后存储到当前目录下,模型名字是protobuf.pbtxt
def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', 'mfnnew.pbtxt', as_text=True)
    return


# 函数功能能,将pb模型转换为txt,转换好后存储到txtmodel文件夹下,模型名字是frozen_model_test.txt
def pb_to_txt(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, "./txtmodel ", 'frozen_model_test.txt', as_text=True)


if __name__ == '__main__':
    #mygraph = pb_to_txt("mfn.pb")
    filepath = 'mfnnew.pb'
    convert_pb_to_pbtxt(filepath)