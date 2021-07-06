from definitions import DATASETS_DIR, MODEL_DIR

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer


# This example is based on: https://www.tensorflow.org/tutorials/text/classify_text_with_bert


class BertExample(object):

    def __init__(self):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.ds_batch_size = 32
        self.ds_seed = 42

        self.train_ds = None
        self.test_ds = None
        self.val_ds = None

        self.classifier_model = None
        pass

    def read_dataset(self):
        dataset_dir = os.path.join(os.path.dirname(DATASETS_DIR))
        train_dir = os.path.join(dataset_dir, 'train')
        test_dir = os.path.join(dataset_dir, 'test')

        # remove unused folders to make it easier to load the data
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size=self.ds_batch_size,
            validation_split=0.2,
            subset='training',
            seed=self.ds_batch_size)

        class_names = raw_train_ds.class_names
        self.train_ds = raw_train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation',
            seed=self.seed)

        self.val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            test_dir,
            batch_size=self.batch_size)

        self.test_ds = test_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        for text_batch, label_batch in self.train_ds.take(1):
            for i in range(3):
                print(f'Review: {text_batch.numpy()[i]}')
                label = label_batch.numpy()[i]
                print(f'Label : {label} ({class_names[label]})')

    def build_classifier_model(self):
        preprocessing_path = os.path.join(MODEL_DIR, 'bert', 'preprocess')
        encoder_path = os.path.join(MODEL_DIR, 'bert', 'encoder')

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(preprocessing_path, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(encoder_path, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        self.classifier_model = tf.keras.Model(text_input, net)

    @staticmethod
    def print_my_examples(inputs, results):
        result_for_printing = \
            [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
             for i in range(len(inputs))]
        print(*result_for_printing, sep='\n')
        print()

    def train(self):
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()

        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        self.classifier_model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=metrics)

        print(f'Training model with bert_en_uncased_L-12_H-256_A-4')
        history = self.classifier_model.fit(x=self.train_ds,
                                            validation_data=self.val_ds,
                                            epochs=epochs)

        loss, accuracy = self.classifier_model.evaluate(self.test_ds)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        saved_model_path = os.path.join(MODEL_DIR, 'inference')

        self.classifier_model.save(saved_model_path, include_optimizer=False)

        reloaded_model = tf.saved_model.load(self.saved_model_path)

        examples = [
            'this is such an amazing movie!',  # this is the same sentence tried earlier
            'The movie was great!',
            'The movie was meh.',
            'The movie was okish.',
            'The movie was terrible...',
            'This movie was just bullshit.',
            'The best movie ever.',
            'LOL the movie was just trash.'
        ]

        reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
        original_results = tf.sigmoid(self.classifier_model(tf.constant(examples)))

        print('Results from the saved model:')
        BertExample.print_my_examples(examples, reloaded_results)
        print('Results from the model in memory:')
        BertExample.print_my_examples(examples, original_results)

