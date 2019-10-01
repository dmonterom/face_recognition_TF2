# from resnet_groupNorm import train_model
from resnet_batchRenorm import train_model
# from resnet import train_model
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

batch_size = 128
reg_coef = 1.0
learning_rate = 0.0015
buffer_size = 100000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # img = tf.image.decode_jpeg(features['image_raw'])
    # img = tf.reshape(img, shape=(112, 112, 3))
    # r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    # img = tf.concat([b, g, r], axis=-1)
    # img = tf.cast(img, dtype=tf.float32)
    # img = tf.subtract(img, 127.5)
    # img = tf.multiply(img, 0.0078125)
    # img = tf.image.random_flip_left_right(img)
    img = features['image_raw']
    label = tf.cast(features['label'], tf.int32)
    return img, label


strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync


with strategy.scope():
    dataset = tf.data.TFRecordDataset(
        'dataset/converted_dataset/ms1m_train.tfrecord')
    dataset_size = sum(1 for _ in dataset)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size).repeat().batch(
        batch_size * num_replicas)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    print("Preparing model...")

    model = train_model()
    # model.load_weights('output/ckpt/ckpt_sgd_5/weights_step-1120000')

    optimizer = tf.keras.optimizers.SGD(
        lr=learning_rate, momentum=0.9, nesterov=False)
    # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    # optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=0.0)


@tf.function
def train_step(_images, _labels, _regCoef):
    def step_fn(images, labels, regCoef):
        with tf.GradientTape() as tape:
            img = tf.image.decode_jpeg(images[0])
            for i in range(batch_size - 1):
                img = tf.concat(
                    [img, tf.image.decode_jpeg(images[i + 1])], axis=0)
            img = tf.reshape(img, shape=(-1, 112, 112, 3))
            r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
            img = tf.concat([b, g, r], axis=-1)
            img = tf.cast(img, dtype=tf.float32)
            img = tf.subtract(img, 127.5)
            img = tf.multiply(img, 0.0078125)
            img = tf.image.random_flip_left_right(img)
            logits = model(img, labels)
            pred = tf.nn.softmax(logits)
            inf_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) * (1.0 / batch_size)
            reg_loss = tf.add_n(model.losses)
            # loss = (inf_loss + reg_loss * regCoef) * (1.0 / num_replicas)
            loss = (inf_loss / regCoef + reg_loss) * (1.0 / num_replicas)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.dtypes.int32), labels), dtype=tf.float32))
        return loss, inf_loss, reg_loss, accuracy
    loss, inf_loss, reg_loss, accuracy = strategy.experimental_run_v2(
        step_fn, args=(_images, _labels, _regCoef,))
    train_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, loss, axis=None)
    inference_loss = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, inf_loss, axis=None)
    regularization_loss = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, reg_loss, axis=None)
    accuracy = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, accuracy, axis=None)
    return accuracy, train_loss, inference_loss, regularization_loss


EPOCHS = 100000

# create log
summary_writer = tf.summary.create_file_writer('output/log')

lr_steps = [int(100000 * 512 / (batch_size * num_replicas)),
            int(180000 * 512 / (batch_size * num_replicas))]

# lr_steps = [
#     int(50000 * 512 / (batch_size * num_replicas)),
#     int(100000 * 512 / (batch_size * num_replicas)),
#     int(150000 * 512 / (batch_size * num_replicas))]

# lr_steps = [int(100000 * 384 / (batch_size * num_replicas)),
#             int(160000 * 384 / (batch_size * num_replicas)),
#             int(200000 * 384 / (batch_size * num_replicas))]

step = 0
epoch = 0
for img, label in dist_dataset:
    step += 1
    epoch = (batch_size * num_replicas * step) // dataset_size
    accuracy, train_loss, inference_loss, regularization_loss = train_step(
        img, label, reg_coef)
    if step % 10 == 0:
        template = 'Epoch {}, Step {}, Loss: {}, Reg loss: {}, Accuracy: {}, Reg coef: {}'
        print(template.format(epoch + 1, step,
                              '%.5f' % (inference_loss),
                              '%.5f' % (regularization_loss),
                              '%.5f' % (accuracy),
                              '%.5f' % (reg_coef)))
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
            optimizer.lr = optimizer.lr * 0.3
    # if inference_loss * 1.0 < regularization_loss * reg_coef:
    #     reg_coef = reg_coef * 0.98
# if epoch + 1 == mod_dataset_epoch:
#     with strategy.scope():
#         dataset = tf.data.TFRecordDataset(
#             'dataset/converted_dataset/ms1m_train.tfrecord')
#         dataset = dataset.map(parse_function)
#         dataset = dataset.shuffle(buffer_size=6000000)
#         dataset = dataset.batch(batch_size * num_replicas)
#         dist_dataset = strategy.experimental_distribute_dataset(dataset)
