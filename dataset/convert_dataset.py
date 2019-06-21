import mxnet as mx
import argparse
import io
import numpy as np
import tensorflow as tf
import os


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='faces_webface_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='faces_webface_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='converted_dataset', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'train.tfrecord')
    writer = tf.data.experimental.TFRecordWriter(output_path)

    def generator():
        for i in imgidx:
            img_info = imgrec.read_idx(i)
            header, img = mx.recordio.unpack(img_info)
            label = int(header.label)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            yield example.SerializeToString()
        # serialized_features_dataset.take(example.SerializeToString())
            if i % 10000 == 0:
                print('%d num image processed' % i)
    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_features_dataset)  # Serialize To String


if __name__ == '__main__':
    # # define parameters
    id2range = {}
    data_shape = (3, 112, 112)
    args = parse_args()
    print("Unpacking mxnet dataset...")
    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)
    print('id2range', len(id2range))

    print("Done.")

    # # generate tfrecords
    print("Generating Tensorflow Dataset...")
    mx2tfrecords(imgidx, imgrec, args)
    print("Done.")
