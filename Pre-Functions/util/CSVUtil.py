import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import os


class CSVUtil:
    filenames = []
    record_defaults = [[1.], [1.]]

    def __init__(self, filenames, record_defaults):
        self.filenames = filenames
        self.record_defaults = record_defaults

    def read_csv(self):
        filename_queue = tf.train.string_input_producer(self.filenames)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        col1, col2 = tf.decode_csv(value, record_defaults=self.record_defaults)

        features = tf.stack([col1, col2])

        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(local_init_op)

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for i in range(30):
                    example, label = sess.run([features, col2])
                    print(example)
                    # print(label)
            except tf.errors.OutOfRangeError:
                print('Done !!!')

            finally:
                coord.request_stop()
                coord.join(threads)


print(os.getcwd())
csv_reader = CSVUtil(['sqrt_minus.csv'], [[1.], [1.]])
csv_reader.read_csv()