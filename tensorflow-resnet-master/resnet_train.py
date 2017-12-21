from resnet import * 
import tensorflow as tf

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
'''
tf.app.flags.DEFINE_string('train_dir', './resnet_train/exp27',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 64000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
'''


def top_1_and_5(predictions, labels):
    #test_size = FLAGS.test_size #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    in_top5 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=5))
    num_correct_1 = tf.reduce_sum(in_top1, name ="top1")
    num_correct_5 = tf.reduce_sum(in_top5, name ="top5")
    return num_correct_1, num_correct_5
    #return (2500 - num_correct_1) / 2500, (2500 - num_correct_5) / 2500

def train(is_training, logits, input_x, labels, sess, images_train, labels_train):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, labels)
    predictions = tf.nn.softmax(logits)

    top1_error, top5_error = top_1_and_5(predictions, labels)



    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    #tf.scalar_summary('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    #tf.scalar_summary('val_top1_error_avg', top1_error_avg)

    #tf.scalar_summary('learning_rate', FLAGS.learning_rate)

    learning_rate = tf.placeholder(tf.float32, shape=[])

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    #opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7)
    grads = opt.compute_gradients(loss_)
    #for grad, var in grads:
        #if grad is not None and not FLAGS.minimal_summaries:
            #tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        #tf.image_summary('images', images)

        #for var in tf.trainable_variables():
            #tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op, name = 'train_op')

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=(FLAGS.max_steps/FLAGS.test_stride)+1)

    testSteps = range(0,FLAGS.max_steps+1,FLAGS.test_stride)[1:]

    #summary_op = tf.merge_all_summaries()
    sess.run(tf.initialize_all_variables())
    sess.graph.finalize()
    tf.train.start_queue_runners(sess=sess)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    """
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    """

    lr = FLAGS.learning_rate

    for x in xrange(FLAGS.max_steps + 1):

        if x == 32000 or x == 48000 or x == 56000:
            lr = lr * 0.1

        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        #write_summary = step % 100 and step > 1
        #if write_summary:
        #    i.append(summary_op)

        #X_batch, Y_batch = next_batch(FLAGS.batch_size, images_train, labels_train)

        #o = sess.run(i, { is_training: True, input_x: X_batch, labels: Y_batch})
        o = sess.run(i, { is_training: True, learning_rate: lr})

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if np.isnan(loss_value):
            continue

        if step % 500 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        #if write_summary:
        #    summary_str = o[2]
        #    summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step in testSteps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        #if step > 1 and step % FLAGS.test_stride == 0:
        #    _, t1, t5 = sess.run([val_op, top1_error, top5_error], { is_training: False })
        #    print("At Step {0}, Top1:{1}, Top5:{2}".format(step,t1,t5))



