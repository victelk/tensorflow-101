#This code classifies handwritten digits 
#Also known as MNIST - Modified National Institute of Standards and Technology database

#This configuration produced 98.01% accuracy for test set whereas it produced 99.77% accuracy for trainset. 
#Producing close accuracy rates is expected for re-run (random initialization causes to produce different results each time)

#blog post: https://sefiks.com/2017/09/11/handwritten-digit-classification-with-tensorflow/

#-----------------------------------------------

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math

tf.logging.set_verbosity(tf.logging.WARN)

#-----------------------------------------------
#variables

epoch = 1500
learningRate = 0.1
batch_size = 120

mnist_data = "/tmp/MNIST_data"

trainForRandomSet = False

#-----------------------------------------------
#data process and transformation

MNIST_DATASET = input_data.read_data_sets(mnist_data)

train_data = np.array(MNIST_DATASET.train.images, 'float32')
train_target = np.array(MNIST_DATASET.train.labels, 'int64')
print("training set consists of ", len(MNIST_DATASET.train.images), " instances")

test_data = np.array(MNIST_DATASET.test.images, 'float32')
test_target = np.array(MNIST_DATASET.test.labels, 'int64')
print("test set consists of ", len(MNIST_DATASET.test.images), " instances")

#-----------------------------------------------
#visualization
print("input layer consists of ", len(MNIST_DATASET.train.images[1]), " features ("
	,math.sqrt(len(MNIST_DATASET.train.images[1])), "x", math.sqrt(len(MNIST_DATASET.train.images[1]))," pixel images)") #28x28 = 784 input feature

#-----------------------------------------------
feature_columns = [tf.contrib.layers.real_valued_column("pixels", dimension=len(MNIST_DATASET.train.images[1]))]

classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns,
    n_classes=10, #0 to 9 - 10 classes
    hidden_units=[128, 32],  #2 hidden layers consisting of 128 and 32 units respectively
    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate),
    activation_fn = tf.nn.relu,
    #activation_fn = tf.nn.softmax
    model_dir="/tmp/model2"
)


'''
def generate_input_fn(data, label):
    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label]
        , batch_size=batch_size
        , capacity=8*batch_size
        , min_after_dequeue=4*batch_size
        , enqueue_many=True
    )
    return {"pixels":image_batch}, label_batch
'''

def my_generate_input_fn(data, label,  perform_shuffle=False):
    dataset = (tf.data.Dataset.from_tensor_slices((data,label)))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size = 256)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return {"pixels":image_batch}, label_batch


def input_fn_for_train():
    return my_generate_input_fn(train_data, train_target, perform_shuffle=True)


def input_fn_for_test():
    return my_generate_input_fn(test_data, test_target)


# To inspect the output of input_fn_for_train:
batch_features, batch_labels = input_fn_for_train()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print((sess.run(batch_features["pixels"])).shape)
    print(batch_features["pixels"].shape)
    print(batch_labels.shape)
    # print(batch_labels[0].eval())
    coord.request_stop()
    coord.join(threads)

'''
#train on small random selected dataset
classifier.train(input_fn=input_fn_for_train, steps=epoch)

print("\n---training is over...")

#----------------------------------------
#apply to make predictions

predictions = list(classifier.predict(input_fn=input_fn_for_test))
predicted_classes = [p["classes"] for p in predictions]
#predicted_classes is list of numpy.ndarray of type class bytes

index = 0
for p in predicted_classes[:10]:

    print("actual: ", test_target[index], ", prediction: ", list(map(lambda x: x.decode('utf-8'),p)))

    pred = MNIST_DATASET.test.images[index]
    pred = pred.reshape([28, 28]);
    plt.gray()
    plt.imshow(pred)
    plt.show()
    index += 1

# ----------------------------------------
# calculationg overall accuracy

print("\n---evaluation...")
accuracy_score = classifier.evaluate(input_fn=input_fn_for_test, steps=epoch)['accuracy']
print("accuracy: ", 100 * accuracy_score, "%")

'''