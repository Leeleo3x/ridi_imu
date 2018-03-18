import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
import os
import models

args = None
nano_to_sec = 1e09


def get_batch(input_feature, input_target, batch_size, num_steps, stride_ratio=1, full_sequence=True, step_size=1):
    total_num, dim = input_feature.shape
    assert input_target.shape[0] == total_num

    partition_length = total_num // batch_size
    feature_batches = np.empty([batch_size, partition_length, dim])
    target_batches = np.empty([batch_size, partition_length, input_target.shape[1]])
    for i in range(batch_size):
        feature_batches[i] = input_feature[i * partition_length:(i + 1) * partition_length, :]
        target_batches[i] = input_target[i * partition_length:(i + 1) * partition_length, :]

    stride = num_steps // stride_ratio
    if full_sequence:
        epoch_size = partition_length // stride
        for i in range(epoch_size):
            if i * stride + num_steps >= feature_batches.shape[1]:
                break
            feat = feature_batches[:, i * stride: i * stride + num_steps, :]
            targ = target_batches[:, i * stride: i * stride + num_steps, :]
            yield (feat, targ)
    else:
        for i in range(0, partition_length-stride, step_size):
            feat = feature_batches[:, i:i+stride, :]
            targ = target_batches[:, i:i+stride, :]
            yield (feat, targ)


def construct_graph(input_dim, output_dim, batch_size=1, softmax=False):
    # construct graph
    init_stddev = 0.001
    # fully_dims = [512, 256, 128]
    fully_dims = [512, 256]
    # placeholders for input and output
    x = tf.placeholder(tf.float32, [batch_size, None, input_dim],
                       name='input_placeholder')
    y = tf.placeholder(tf.float32, [batch_size, None, output_dim],
                       name='output_placeholder')

    cell = tf.contrib.rnn.BasicLSTMCell(args.state_size, state_is_tuple=True)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * args.num_layer, state_is_tuple=True)
    init_state = tf.placeholder(tf.float32, [args.num_layer, 2, None, args.state_size], name='init_state')
    l = tf.unstack(init_state)
    state_tuple = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
         for idx in range(args.num_layer)])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, x, initial_state=state_tuple)
    final_state = tf.identity(final_state, name='final_state')

    # Fully connected layer

    with tf.variable_scope('fully_connected1'):
        W1 = tf.get_variable('W1', shape=[args.state_size, fully_dims[0]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        b1 = tf.get_variable('b1', shape=[fully_dims[0]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        out_fully1 = tf.tanh(tf.matmul(tf.reshape(rnn_outputs, [-1, args.state_size]), W1) + b1)
    with tf.variable_scope('fully_connected2'):
        W2 = tf.get_variable('W2', shape=[fully_dims[0], fully_dims[1]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        b2 = tf.get_variable('b2', shape=[fully_dims[1]])
        out_fully2 = tf.tanh(tf.matmul(out_fully1, W2) + b2)
    # with tf.variable_scope('fully_connected3'):
    #     W3 = tf.get_variable('W3', shape=[fully_dims[1], fully_dims[2]],
    #                          initializer=tf.random_normal_initializer(stddev=init_stddev))
    #     b3 = tf.get_variable('b3', shape=[fully_dims[2]],
    #                          initializer=tf.random_normal_initializer(stddev=init_stddev))
    #     out_fully3 = tf.tanh(tf.matmul(out_fully2, W3) + b3)

    # output layer
    with tf.variable_scope('output_layer'):
        W = tf.get_variable('W', shape=[fully_dims[-1], output_dim])
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.random_normal_initializer(stddev=init_stddev))
    # regressed = tf.matmul(tf.reshape(rnn_outputs, [-1, args.state_size]), W) + b
    regressed = tf.matmul(out_fully2, W) + b
    if softmax:
        regressed = tf.nn.softmax(regressed)
    regressed = tf.identity(regressed, name='regressed')
    return {'x': x, 'y': y, 'init_state': init_state, 'final_state': final_state, 'regressed': regressed}


def run_testing(sess, variable_dict, feature, target, init_state):
    input_dim, output_dim = feature.shape[1], target.shape[1]

    feature_rnn = feature.reshape([1, -1, input_dim])
    target_rnn = target.reshape([1, -1, output_dim])
    regressed_rnn, loss = sess.run([variable_dict['regressed'], variable_dict['total_loss']],
                                   feed_dict={variable_dict['x']: feature_rnn, variable_dict['y']: target_rnn,
                                              variable_dict['init_state']: init_state})
    predicted = np.array(regressed_rnn).reshape([-1, output_dim])
    return predicted, loss


def log(path, *args):
    if path is None:
        print(args)
    else:
        with open(path, 'a') as f:
            f.write(" ".join(args))
            f.write('\n')


def run_training(model, num_epoch, verbose=True, output_path=None, tensorboard_path=None, checkpoint_path=None,
                 log_path=None):
    tf.reset_default_graph()
    # construct graph
    variable_dict = construct_graph(model.input_dim(), model.output_dim(), args.batch_size, model.softmax)
    x = variable_dict['x']
    y = variable_dict['y']
    regressed = variable_dict['regressed']
    init_state = variable_dict['init_state']
    final_state = variable_dict['final_state']

    # loss and training step
    total_loss = tf.reduce_mean(
        tf.squared_difference(tf.reshape(regressed, [-1, model.output_dim()]), tf.reshape(y, [-1, model.output_dim()])))
    variable_dict['total_loss'] = total_loss

    tf.summary.scalar('mean squared loss', total_loss)
    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('rnn_input', x)
    tf.add_to_collection('rnn_output', y)
    tf.add_to_collection('regressed', regressed)
    tf.add_to_collection('state_size', args.state_size)
    tf.add_to_collection('num_layer', args.num_layer)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step, args.decay_rate)

    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    all_summary = tf.summary.merge_all()

    report_interval = 100
    global_counter = 0
    with tf.Session() as sess:
        train_writer = None
        if tensorboard_path is not None:
            train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        sess.run(tf.global_variables_initializer())
        training_losses = []
        validation_losses = []
        saver = None
        if output_path is not None:
            saver = tf.train.Saver()
        for i in range(num_epoch):
            if verbose:
                log(log_path, 'EPOCH %d' % i)
            epoch_loss = 0.0
            steps_in_epoch = 0
            for features, targets in model.training_data():
                state = np.zeros((args.num_layer, 2, args.batch_size, args.state_size))
                for _, (X, Y) in enumerate(get_batch(features, targets,
                                                     args.batch_size, args.num_steps, full_sequence=model.full_sequence,
                                                     step_size=args.num_steps//20)):
                    summaries, current_loss, state, _ = sess.run([all_summary,
                                                                  total_loss,
                                                                  final_state,
                                                                  train_step],
                                                                 feed_dict={x: X, y: Y, init_state: state})
                    if not model.full_sequence:
                        state = np.zeros((args.num_layer, 2, args.batch_size, args.state_size))
                    epoch_loss += current_loss
                    if (checkpoint_path is not None) and global_counter % args.checkpoint == 0 and global_counter > 0:
                        saver.save(sess, os.path.join(checkpoint_path, 'ckpt'), global_step=global_counter)
                        log(log_path, 'Checkpoint file saved at step', str(global_counter))
                    if global_counter % report_interval == 0 and global_counter > 0 and verbose:
                        if tensorboard_path is not None:
                            train_writer.add_summary(summaries, global_counter)
                    steps_in_epoch += 1
                    global_counter += 1
            training_losses.append(epoch_loss / steps_in_epoch)
            log(log_path,
                'Training loss at epoch {:d} (step {:d}): {:f}'.format(i, global_counter, training_losses[-1]))

            # run validation
            predicted_concat = []
            target_concat = []
            for valid_feature, valid_target in model.validation_data():
                state = np.zeros((args.num_layer, 2, 1, args.state_size))
                results = []
                for _, (X, Y) in enumerate(get_batch(valid_feature, valid_target,
                                                     args.batch_size, args.num_steps, full_sequence=model.full_sequence,
                                                     step_size=args.num_steps)):
                    predicted, cur_loss = sess.run([variable_dict['regressed'], variable_dict['total_loss']],
                                                   feed_dict={variable_dict['x']: X, variable_dict['y']: Y,
                                                              variable_dict['init_state']: state})
                    results.append(predicted)
                    if not model.full_sequence:
                        state = np.zeros((args.num_layer, 2, args.batch_size, args.state_size))
                # loss_sklearn = mean_squared_error(np.reshape(np.array(predicted), [-1, 2]), valid_targets[valid_id])
                # print('Loss for valid set {}: {:.6f}(tf), {:.6f}(sklearn)'.format(valid_id, cur_loss, loss_sklearn))
                predicted_concat.append(np.concatenate(results, axis=0))
                target_concat.append(valid_target)
            predicted_concat = np.concatenate(predicted_concat, axis=0)
            target_concat = np.concatenate(target_concat, axis=0)
            l2_loss = np.array([mean_squared_error(predicted_concat[:, i], target_concat[:, i])
                                for i in range(predicted_concat.shape[1])])
            validation_losses.append(l2_loss)
            log(log_path, 'Validation loss at epoch {:d} (step {:d}):'.format(i, global_counter),
                str(validation_losses[-1]),
                str(np.average(l2_loss)))

        if output_path is not None:
            saver.save(sess, output_path, global_step=global_counter)
            log(log_path, 'Meta graph saved to', output_path)

        # output final training loss
        # total_samples = 0
        # train_error_axis = np.zeros(output_dim, dtype=float)
        # for data_id in range(len(features)):
        #     feature_rnn = features[data_id].reshape([1, -1, input_dim])
        #     target_rnn = targets[data_id].reshape([1, -1, output_dim])
        #     predicted = sess.run([regressed], feed_dict={x: feature_rnn, y: target_rnn})
        #     diff = np.power(np.array(predicted).reshape([-1, output_dim]) - targets[data_id], 2)
        #     train_error_axis += np.sum(diff, axis=0)
        #     total_samples += features[data_id].shape[0]
        # print('Overall training loss:', train_error_axis / total_samples)

    return training_losses, validation_losses


if __name__ == '__main__':
    import pandas
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list', type=str)
    parser.add_argument('validation', type=str)
    parser.add_argument('--target', type=str, default="velocity")
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
    parser.add_argument('--use_pdf', action='store_true', default=False)
    args = parser.parse_args()

    # configure output path
    output_root = None
    model_path = None
    tfboard_path = None
    chpt_path = None
    log_path = None
    if args.output is not None:
        output_root = args.output
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        model_path = output_root + '/model.tf'
        tfboard_path = output_root + '/tensorboard'
        chpt_path = output_root + '/checkpoints/'
        log_path = os.path.join(output_root, 'log.txt')
        if not os.path.exists(tfboard_path):
            os.makedirs(tfboard_path)
        if not os.path.exists(chpt_path):
            os.makedirs(chpt_path)

    if args.target == 'angle':
        model = models.AngleModel(args.list, args.validation)
    elif args.target == 'velocity':
        model = models.VelocityModel(args.list, args.validation, args.feature_smooth_sigma, args.target_smooth_sigma)
    elif args.target == 'position':
        model = models.PositionModel(args.list, args.validation)
    else:
        assert "Unreachable"

    print('Running training')
    training_losses, validation_losses = run_training(model, args.num_epoch, output_path=model_path,
                                                      tensorboard_path=tfboard_path, checkpoint_path=chpt_path,
                                                      log_path=log_path)

    if output_root is not None:
        assert len(training_losses) == len(validation_losses)
        with open(output_root + '/losses.txt', 'w') as f:
            for i in range(len(training_losses)):
                f.write('{} {} {}\n'.format(training_losses[i], validation_losses[i][0], validation_losses[i][1]))

    # plt.plot(losses)
    # plt.show()
