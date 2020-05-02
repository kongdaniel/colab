import tensorflow as tf
modelPb="./mfn.pb"
saveFile="./saveFile"
gf = tf.GraphDef()
m_file = open(modelPb, 'rb')
gf.ParseFromString(m_file.read())

with open(saveFile, 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name + '\n')

file = open(saveFile, 'r')
data = file.readlines()
print("output name = " + data[len(data) - 1])

print("Input name = ")
file.seek(0)
print(file.readline())