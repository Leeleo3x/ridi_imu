import argparse
import pandas
import os
import rnn
import tensorflow as tf
import numpy as np
from write_trajectory_to_ply import write_ply_to_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--model', type=str, default='speed')
    parser.add_argument('--feature_smooth_sigma', type=float, default=-1.0)
    parser.add_argument('--target_smooth_sigma', type=float, default=30.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--state_size', type=int, default=1500)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay_step', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=5000)
    return parser.parse_args()


def load_dataset(path, imu_columns):
    data_all = pandas.read_csv(os.path.join(path, 'processed', 'data.csv'))
    feature_vectors = data_all[imu_columns].values
    return feature_vectors


def predict(checkpoint, meta_graph, feature_vectors):
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        state = tuple(
            [(np.zeros([1, 1500]), np.zeros([1, 1500]))
             for i in range(1)])
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input_placeholder:0')
        # init_state = graph.get_tensor_by_name('init_state')
        regressed = graph.get_tensor_by_name('regressed:0')
        predicted = [np.array([0, 0, 0])]
        for i in range(features.shape[0]):
            X = np.concatenate([feature_vectors[i], predicted[-1]]).reshape([1, -1, 12])
            result = sess.run(regressed, feed_dict={
                x: X,
                # init_state: state
            })
            predicted.append(result[0])
        return predicted


def main():
    args = parse_args()
    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z', 'grav_x', 'grav_y', 'grav_z']
    features = load_dataset(args.path, imu_columns)
    predict(args.checkpoint, ',', features)


if __name__ == '__main__':
    args = parse_args()
    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z', 'grav_x', 'grav_y', 'grav_z']
    features = load_dataset(args.path, imu_columns)
    predicted = predict(args.checkpoint, os.path.join(args.checkpoint, 'ckpt-15000.meta'), features)
    result_folder = os.path.join(args.path, 'results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    write_ply_to_file(os.path.join(result_folder, 'rnn.ply'), np.array(predicted))
    # main()
