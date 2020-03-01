import tensorflow as tf
from project_config import BAD_IMAGE_FILENAMES

class dataset:

    def __init__(self,
                 batch_size,
                 image_dim,
                 file_path):

        self.batch_size = batch_size
        self.image_dim = image_dim
        self.file_path = file_path

    @tf.function
    def __preprocessing_image__(self, fn):
        img = tf.io.read_file(fn)
        img = tf.image.decode_png(img)
        img = tf.image.resize(img, (self.image_dim[0], self.image_dim[1]), 'bicubic')
        img = self.__argumentation__(img)
        return (tf.cast(img, tf.float32) - 127.5) / 127.5

    @tf.function
    def __argumentation__(self, image):
        image = tf.image.random_flip_left_right(image)
        return image

    def remove_bad_files(self, data_filenames):
        for fn in BAD_IMAGE_FILENAMES:
            full_path = os.path.join(self.file_path, fn)
            if full_path in data_filenames:
                data_filenames.remove(os.path.join(self.file_path, fn))
        return data_filenames

    def GetDataset(self):
        data_filenames = glob(os.path.join(self.file_path, '*.png'))
        data_filenames = self.remove_bad_files(data_filenames)
        self.data_len = len(data_filenames)
        self.ds = tf.data.Dataset.from_tensor_slices(data_filenames).shuffle(self.data_len)
        self.ds = self.ds.apply(tf.data.experimental.shuffle_and_repeat(self.data_len))
        self.ds = self.ds.apply(tf.data.experimental.map_and_batch(self.__preprocessing_image__, self.batch_size))
        return iter(self.ds)

    def __len__(self):
        return np.ceil(self.data_len/self.batch_size)
