# from resnet_groupNorm import train_model
from resnet_batchRenorm import train_model
# from resnet import train_model
import tensorflow as tf
import random

batch_size = 16
batch_multiplier = 6


def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int32)
    return img, label


dataset = tf.data.TFRecordDataset(
    'dataset/converted_dataset/ms1m_train.tfrecord')
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(buffer_size=20000)
dataset = dataset.batch(batch_size * batch_multiplier)

print("Preparing model...")

model = train_model()

learning_rate = 0.0005  # 0.0005
optimizer = tf.keras.optimizers.SGD(
    lr=learning_rate, momentum=0.9, nesterov=False)
# optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
# optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=0.0)
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(
#     name='train_accuracy')
# inference_loss = tf.keras.metrics.Mean(name='inference_loss')
# regularization_loss = tf.keras.metrics.Mean(
#    name='regularization_loss')


@tf.function
def train_step(images, labels):
    # print(images, labels)
    with tf.GradientTape() as tape:
        logits = model(tf.slice(images, [0, 0, 0, 0], [
                       batch_size, 112, 112, 3]), tf.slice(labels, [0], [batch_size]))
        for i in range(batch_multiplier - 1):
            logits = tf.concat([logits, model(tf.slice(images, [batch_size * (i + 1), 0, 0, 0], [
                               batch_size, 112, 112, 3]), tf.slice(labels, [batch_size * (i + 1)], [batch_size]))], 0)
        pred = tf.nn.softmax(logits)
        # epsilon = tf.constant(value=0.00001)
        # logits = logits + epsilon
        inf_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        reg_loss = tf.add_n(model.losses)
        loss = inf_loss + reg_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # train_loss(loss)
    train_loss = tf.reduce_mean(loss)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.dtypes.int32), labels), dtype=tf.float32))
    inference_loss = tf.reduce_mean(inf_loss)
    regularization_loss = tf.reduce_mean(reg_loss)
    # inference_loss(inf_loss)
    # regularization_loss(reg_loss)
    return accuracy, train_loss, inference_loss, regularization_loss


EPOCHS = 100000

# create log
summary_writer = tf.summary.create_file_writer('output/log')

lr_steps = [int(40000 * 512 / (batch_size * batch_multiplier)),
            int(60000 * 512 / (batch_size * batch_multiplier)),
            int(80000 * 512 / (batch_size * batch_multiplier)),
            int(120000 * 512 / (batch_size * batch_multiplier))]
print(lr_steps)
step = 0
for epoch in range(EPOCHS):
    iterator = iter(dataset)
    while True:
        img, label = next(iterator)
        if (img.shape[0] != batch_size * batch_multiplier or img.shape[0] != label.shape[0]):
            print("End of epoch {}".format(epoch + 1))
            break
        step += 1
        accuracy, train_loss, inference_loss, regularization_loss = train_step(
            img, label)
        if step % 10 == 0:
            template = 'Epoch {}, Step {}, Loss: {}, Reg loss: {}, Accuracy: {}'
            print(template.format(epoch + 1, step,
                                  inference_loss,
                                  regularization_loss,
                                  accuracy))
            with summary_writer.as_default():
                tf.summary.scalar(
                    'train loss', train_loss, step=step)
                tf.summary.scalar(
                    'inference loss', inference_loss, step=step)
                tf.summary.scalar(
                    'regularization loss', regularization_loss, step=step)
                tf.summary.scalar(
                    'train accuracy', accuracy, step=step)
                tf.summary.scalar(
                    'learning rate', optimizer.lr, step=step)
                # for i in range(len(gradients)):
                #     gradient_name = model.trainable_variables[i].name
                #     tf.summary.histogram(
                #         gradient_name + '/gradient', gradients[i], step=step)
                # for weight in model.trainable_variables:
                #     tf.summary.histogram(
                #         weight.name, weight, step=step)
                # layer_output = model.get_layer('').output
                # tf.summary.histogram('name', layer_output)
        if step % 4000 == 0 and step > 0:
            model.save_weights(
                'output/ckpt/weights_step-{}'.format(step))
        for lr_step in lr_steps:
            if lr_step == step:
                optimizer.lr = optimizer.lr * 0.5
