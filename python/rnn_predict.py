import argparse
import pandas
import os
import models
import tensorflow as tf
import numpy as np
from write_trajectory_to_ply import write_ply_to_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--target', type=str, default='velocity')
    return parser.parse_args()


def load_dataset(path, imu_columns):
    data_all = pandas.read_csv(os.path.join(path, 'processed', 'data.csv'))
    feature_vectors = data_all[imu_columns].values
    return feature_vectors


def predict(checkpoint, meta_graph, model):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input_placeholder:0')
        init_state = graph.get_tensor_by_name('init_state:0')
        regressed = graph.get_tensor_by_name('regressed:0')
        final_state = graph.get_tensor_by_name('final_state:0')
        predicted = [np.array([0, 0, 0])]
        state = np.zeros((1, 2, 1, 1500))
        trajectory = []
        for features, targets in model.training_data():
            # X = np.concatenate([feature_vectors[i], predicted[-1]]).reshape([1, -1, 12])
            result, state = sess.run([regressed, final_state], feed_dict={
                x: features.reshape(1, -1, 9),
                init_state: state
            })
            return model.trajectory_from_prediction(result)


def main():
    args = parse_args()
    if args.target == 'angle':
        model = models.AngleModel(args.path)
    elif args.target == 'velocity':
        model = models.VelocityModel(args.path)
    file_name = open(os.path.join(args.checkpoint, 'checkpoint')).readline().split(':')[1].replace('"', '').strip()
    predicted = predict(args.checkpoint, file_name+'.meta', model)
    result_folder = os.path.join(args.path, 'results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    write_ply_to_file(os.path.join(result_folder, 'rnn.ply'), np.array(predicted))


if __name__ == '__main__':
    main()
