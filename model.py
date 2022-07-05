import os
import tensorflow_hub as hub
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from shelve_db import ModelShelveDb
import tensorflow_text as text

"""
Model - Class for the currently used model.

parameters:
model_shelve_db = saves an object of model's shelve data base
url_for_embedding: Dictionaries for intermediate_layer download URLs.
intermediate_layers: pDictionaries for the wanted intermediate_layer, 768 is the hidden size of embedding sequence_output.
model: Reference to the model (tf.keras.Model)
embedding_preprocess: Reference to embedding preprocess
embedding_encoder: Reference to embedding encoder
update_epochs: Slot for pyqtSignal
update_summary: # Slot for pyqtSignal
"""


class Model:
    def __init__(self, slot_for_epochs, slot_for_summary):
        super(Model, self).__init__()
        model_shelve_db = ModelShelveDb()
        # Dictionaries for intermediate_layer download URLs.
        self.url_for_embedding = model_shelve_db.get_url_for_embedding()
        # Dictionaries for the wanted intermediate_layer, 768 is the hidden size of embedding sequence_output.
        self.intermediate_layers = {
            "gru": tf.keras.layers.GRU(768),
            "lstm": tf.keras.layers.LSTM(768)
        }
        self.model = None
        self.embedding_preprocess = None
        self.embedding_encoder = None
        self.update_epochs = slot_for_epochs
        self.update_summary = slot_for_summary
        self.lambda_callback = LambdaCallback(on_epoch_end=self.print_custom)

    """
    Load pretrained ambedding from tensorflow hub
    """
    def load_embedding(self, embedding_name):
        self.embedding_preprocess = hub.KerasLayer(self.url_for_embedding[embedding_name][0])
        self.embedding_encoder = hub.KerasLayer(self.url_for_embedding[embedding_name][1])

    """
    Assemble the language model
    """
    def build_model(self, intermediate_layer, output_vector_size, dropout_value=0.4,
                    activation_function_for_output_layer='softmax', learning_rate=0.0001,
                    loss_function='categorical_crossentropy'):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_layer")
        preprocessed_text = self.embedding_preprocess(text_input)
        encoder_output = self.embedding_encoder(preprocessed_text)

        # "sequence_output": representations of every token in the input sequence with
        # shape [batch size, max sequence length, hidden size(768)].
        layer = self.intermediate_layers[intermediate_layer](encoder_output['sequence_output'])
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Dropout(dropout_value, name='dropout')(layer)
        layer = tf.keras.layers.Dense(output_vector_size, activation=activation_function_for_output_layer,
                                      name='output')(layer)
        self.model = tf.keras.Model(inputs=[text_input], outputs=[layer])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        short_model_summary = "\n".join(summary)
        self.update_summary.emit(short_model_summary)

    """
    Target Task Language Model Fine Tuning.
    This method is the second step of the ULMFiT algorithm.
    The purpose of the second stage is to fine tune the model on a dataset that is related to the target task.
    """
    def target_task_language_model_fine_tuning(self, model_dir_name, intermediate_layer, total_unique_words,
                                               sequences_train, labels_train, needed_epochs, embedding_name):
        train_history = None
        if model_dir_name is not None and os.path.isdir(model_dir_name):  # Checks if tuned model is exist.
            self.model = tf.keras.models.load_model(model_dir_name)  # If yes,load weights of the tuned model.
        else:  # Else, tuned the model.
            self.load_embedding(embedding_name)
            self.build_model(intermediate_layer, total_unique_words)
            train_history = self.model.fit(sequences_train, labels_train, epochs=needed_epochs,
                                           callbacks=[self.lambda_callback])
        return self.model, train_history

    def fine_tuned_the_model(self, model_dir_name, number_of_target_class_classes, x_train, y_train, x_test, y_test):
        new_layers = [tf.keras.layers.Dense(64, activation='relu', name='relu'),
                      tf.keras.layers.Dense(number_of_target_class_classes, activation='softmax', name='softmax')]
        is_model_loaded = self.adjust_model_architecture_for_target_task(new_layers, model_dir_name)
        if not is_model_loaded:
            fine_tuning_history = self.target_task_classifier_fine_tuning(x_train, y_train, x_test,
                                                                          y_test, needed_epochs=20)

            return fine_tuning_history

    """
    Adjust the language model according to the targeted task.
    This method is adjusting the model architecture according to the targeted task.
    At this stage, the model is predicting the next word in a sequence in the context of the data of the target task.
    Afterward, in preparation for the third stage of the ULMFiT algorithm, the model must be adjusted according 
    to the targeted task.
    First the SoftMax layers are cut off, then Two linear blocks are added at the end of the model. 
    Each block uses dropout and batch normalization, with ReLU activation function for the first linear layer and 
    a SoftMax activation that outputs a probability distribution over target classes at the last layer.
    """
    def adjust_model_architecture_for_target_task(self, new_layers, model_dir_name, dropout_value=0.4,
                                                  learning_rate=0.0001, loss_function='categorical_crossentropy'):
        is_model_loaded = False
        if os.path.isdir(model_dir_name):  # Checks if tuned model is exist.
            self.model = tf.keras.models.load_model(model_dir_name)  # If yes,load weights of the tuned model.
            is_model_loaded = True
        else:  # Else, tuned the model.
            last_layer = self.model.layers[3].output
            for layer in new_layers:
                last_layer = tf.keras.layers.BatchNormalization()(last_layer)
                last_layer = tf.keras.layers.Dropout(dropout_value)(last_layer)
                last_layer = layer(last_layer)
            self.model = tf.keras.Model(inputs=[self.model.layers[0].input], outputs=[last_layer])
            self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function,
                               metrics=['accuracy'])
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        short_model_summary = "\n".join(summary)
        self.update_summary.emit(short_model_summary)
        return is_model_loaded

    """
    Language model fine tuning on the target task.
    This method is the third step of the ULMFiT algorithm.
    The model is frozen and then it is gradually unfrozen to fine tune it.
    """
    def target_task_classifier_fine_tuning(self, x_train, y_train, x_val, y_val, needed_epochs):
        self.model.layers[
            3].trainable = False  # freeze the intermediate layer (LSTM or GRU) to avoid catastrophic forgetting
        self.model.layers[4].trainable = False  # freeze the first linear block to avoid catastrophic forgetting
        self.model.fit(x_train, y_train, epochs=5, callbacks=[self.lambda_callback])  # fine-tuned
        self.model.layers[4].trainable = True  # first linear block layer is unfrozen
        self.model.fit(x_train, y_train, epochs=5, callbacks=[self.lambda_callback])  # fine-tuned
        self.model.layers[3].trainable = True  # Intermediate layer (LSTM or GRU) is unfrozen
        fine_tuning_history = self.model.fit(x_train, y_train, epochs=needed_epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[self.lambda_callback])  # fine-tuned
        return fine_tuning_history

    """
    This method is replacing the regular printing method of keras.model.fit
    and with pyqtsignal it directed to QTextEdit in the build new model language screen and main screen to show 
    to user the training and fine tuning process
    """
    def print_custom(self, epoch, logs):
        self.update_epochs.emit(f"\nEpoch number {epoch + 1} \nScore {logs}\n")
