from resnet import train_model
import tensorflow as tf
import random

batch_size = 16
batch_multiplier = 8


def parse_dataset():
    raw_image_dataset = tf.data.TFRecordDataset(
        'dataset/converted_dataset/train.tfrecord')

    image_feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    x_train = []
    y_train = []
    i = 0
    for image_features in parsed_image_dataset:
        i += 1
        if (i % 1000 == 0):
            print("%d dataset images converted" % i)
        x_train.append(image_features['image_raw'])
        y_train.append(image_features['label'])
        # if (i == 1024):
        #     return x_train, y_train, i
    return x_train, y_train, i


print("Preparing dataset...")

x_train, y_train, length = parse_dataset()

print("Done.")

print("Preparing model...")

model = train_model()

# loss_object = tf.keras.losses.CategoricalCrossentropy()
learning_rate = 0.1
optimizer = tf.keras.optimizers.SGD(
    lr=learning_rate, decay=0.0, momentum=0.9, nesterov=False)
# optimizer = tf.keras.optimizers.Adam(
#     lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=0.0)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(
    name='train_accuracy')
inference_loss = tf.keras.metrics.Mean(name='inference_loss')
regularization_loss = tf.keras.metrics.Mean(
    name='regularization_loss')


@tf.function
def train_step(images, labels):
    # print(images, labels)
    with tf.GradientTape() as tape:
        logits = model(tf.slice(images, [0, 0, 0, 0], [
                       16, 112, 112, 3]), tf.slice(labels, [0], [16]))
        for i in range(batch_multiplier - 1):
            logits = tf.concat([logits, model(tf.slice(images, [16*(i+1), 0, 0, 0], [
                       16, 112, 112, 3]), tf.slice(labels, [16*(i+1)], [16]))], 0)
        pred = tf.nn.softmax(logits)
        # y = tf.one_hot(labels, depth=10572)
        # inf_loss = loss_object(y_true=y, y_pred=pred)
        inf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        reg_loss = model.losses
        loss = inf_loss + reg_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    for i in range(len(gradients)):
        gradients[i] /= 10572
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    # train_accuracy(
    #     tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    inference_loss(inf_loss)
    regularization_loss(reg_loss)
    return gradients, accuracy


EPOCHS = 100000

batch_idx = 0
list_img = []
list_label = []
step = 0
# shufle dataset
z = list(zip(x_train, y_train))
random.shuffle(z)
x_train, y_train = zip(*z)

# create log
summary_writer = tf.summary.create_file_writer('output/logs')

lr_steps = [100000 * 512 / (batch_size*batch_multiplier), 140000
            * 512 / (batch_size*batch_multiplier), 160000 * 512 / (batch_size*batch_multiplier)]

lower_loss = 18.2
loss_i = 0

for epoch in range(EPOCHS):
    for i in range(len(x_train)):
        img = tf.image.decode_jpeg(x_train[i])
        img = tf.reshape(img, shape=(1, 112, 112, 3))
        r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
        img = tf.concat([b, g, r], axis=-1)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.subtract(img, 127.5)
        img = tf.multiply(img, 0.0078125)
        img = tf.image.random_flip_left_right(img)
        label = tf.cast(y_train[i], tf.int64)
        if (batch_idx == 0):
            list_img = []
            list_label = []
        list_img.append(img)
        list_label.append(label)
        batch_idx += 1
        if (batch_idx == batch_size * batch_multiplier):
            step += 1
            batch_idx = 0
            batch_img = tf.concat(list_img, axis=0)
            batch_label = tf.stack(list_label, axis=0)
            gradients, accuracy = train_step(
                batch_img, batch_label)
            if step % 10 == 0:
                template = 'Epoch {}, Step {}, Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1, step,
                                      train_loss.result(),
                                      accuracy))
                with summary_writer.as_default():
                    tf.summary.scalar(
                        'train loss', train_loss.result(), step=step)
                    tf.summary.scalar(
                        'inference loss', inference_loss.result(), step=step)
                    tf.summary.scalar(
                        'regularization loss', regularization_loss.result(), step=step)
                    tf.summary.scalar(
                        'train accuracy', accuracy, step=step)
                    tf.summary.scalar(
                        'learning rate', learning_rate, step=step)
                    for i in range(len(gradients)):
                        gradient_name = model.trainable_variables[i].name
                        tf.summary.histogram(
                            gradient_name + '/gradient', gradients[i], step=step)
            for lr_step in lr_steps:
                if lr_step == step:
                    optimizer.lr = optimizer.lr * 0.1
    model.save_weights(
        'output/ckpt/weights_epoch-{}'.format(epoch + 1))
