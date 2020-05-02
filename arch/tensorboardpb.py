import tensorflow as tf
from tensorflow.python.platform import gfile
model = './mfn.pb' #pb文件名称
model1 ='./ckpt_best/mobilefacenet_model.pbfrozen_model.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()

#graph_def.SerializeToOstream(tf.gfile.FastGFile(model, 'rb').read())
graph_def.ParseFromString(tf.gfile.GFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('./logpb/', graph)
#log存放地址
# import tensorflow as tf
# path ='./ckpt_best/mobilefacenet_model.pb'
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(path, 'rb').read())
#
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('/home/aldy/log/', graph)