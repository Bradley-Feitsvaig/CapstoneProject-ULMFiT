import os
import sys

from threading import Thread
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QMovie, QPixmap, QTextCursor
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QStackedWidget, QFileDialog
from sklearn.model_selection import train_test_split
from dataset_preprocess import DatasetPreprocess
from misc_methods import MiscMethods
from model import Model
import codecs
from shelve_db import AppShelveDb

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

"""
WelcomeScreen - controller of the Welcome screen
"""


class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi(ROOT_DIR + "/ui/welcomeScreen.ui", self)
        self.start_button.clicked.connect(self.goto_main_screen)
        self.user_help_button.clicked.connect(self.open_help_file)

    def goto_main_screen(self):  # Switch screen to main screen
        widget.setCurrentIndex(widget.currentIndex() + 1)
        widget.setFixedHeight(1000)
        widget.setFixedWidth(1300)

    def open_help_file(self):
        os.startfile(ROOT_DIR + "/ui/userHelp.txt")



"""
MainScreen - controller of the Welcome screen

parameters:
current_model = saves an object of the current used model in MainScreen
movie = saves the gif file for loading animation
app_shelve_db = saves an object of application's shelve data base
model_task_dict = dictionary of trained model and their meta data, for example:
            'English with LSTM wiki2': {'tasks': ['Sentiment analysis'],
                                        'base language model': "\\savedModels\\englishLSTMNextWord",
                                        'Sentiment analysis model': "\\savedModels\\englishLSTMsentimental",
                                        'Sentiment analysis dataset': "\\datasets\\Twitter US Airline Sentiment "
                                                                      "dataset.csv",
                                        'language': 'english'
                                        },
            'English with GRU wiki2': {'tasks': ['Sentiment analysis'],
                                       'base language model': "\\savedModels\\englishGRUNextWord",
                                       'Sentiment analysis model': "\\savedModels\\englishGRUsentimental",
                                       'Sentiment analysis dataset': "\\datasets\\Twitter US Airline Sentiment "
                                                                     "dataset.csv",
                                       'language': 'english'
                                       },
file_name = A path to the last file selected by the user on which the prediction will be performed
task_data_set_file_name = A path to the last dataset selected by the user on which the language model will be fine tuned
"""


class MainScreen(QDialog):
    fine_tuning_text_update = pyqtSignal(str)

    def __init__(self):
        super(MainScreen, self).__init__()
        self.current_model = None
        loadUi(ROOT_DIR + "/ui/mainScreen.ui", self)
        self.movie = QMovie(ROOT_DIR + "/img/loading.gif")
        self.return_to_welcome.clicked.connect(self.goto_welcome_screen)
        self.new_language_model_button.clicked.connect(self.goto_build_new_model_screen)
        self.fine_tune_the_model_button.clicked.connect(self.fine_tuned_the_model_clicked)
        self.LM_comboBox.activated.connect(self.lm_combobox_activated)
        self.task_comboBox.activated.connect(self.task_combobox_activated)
        self.app_shelve_db = AppShelveDb()
        self.model_task_dict = self.app_shelve_db.get_model_task_dict()
        self.LM_comboBox.addItems(self.model_task_dict.keys())
        self.load_text_file_radioButton.clicked.connect(self.change_load_text_status)
        self.insert_text_manually_radioButton.clicked.connect(self.change_load_text_status)
        self.load_text_file_button.clicked.connect(self.open_file_name_dialog)
        self.choose_tasks_dataset_button.clicked.connect(self.open_file_choose_task_data_set_name_dialog)
        self.predict_button.clicked.connect(self.get_prediction_clicked)
        self.file_name = None
        self.task_data_set_file_name = None
        self.fine_tuning_text_cursor = self.fine_tuning_text.textCursor()
        self.fine_tuning_text_update.connect(lambda value: self.change_fine_tuning_text(value))
        self.clear_page_button.clicked.connect(self.refresh_page)

    def refresh_page(self):  # Refreshes all parameters in the page
        self.file_name = None
        self.task_data_set_file_name = None
        self.current_model = None
        self.refresh_model_task_dict()
        self.fine_tune_the_model_button.setEnabled(False)
        self.load_text_file_button.setEnabled(False)
        self.lineEdit.setEnabled(False)
        self.lineEdit.clear()
        self.predict_button.setEnabled(False)
        self.insert_text_manually_radioButton.setChecked(False)
        self.insert_text_manually_radioButton.setEnabled(False)
        self.load_text_file_radioButton.setChecked(False)
        self.load_text_file_radioButton.setEnabled(False)
        self.fine_tuning_text.clear()
        self.place_for_prediction_plot_label.clear()

    def refresh_model_task_dict(
            self):  # Refreshes the model_task_dict in case of changes such as new language model build
        self.model_task_dict = self.app_shelve_db.get_model_task_dict()
        self.LM_comboBox.clear()
        self.LM_comboBox.addItems(self.model_task_dict.keys())
        self.lm_combobox_activated()

    def change_fine_tuning_text(self, value):  # Method for print in QTextEdit in the right bottom of the screen
        self.fine_tuning_text_cursor.movePosition(QTextCursor.End)
        self.fine_tuning_text.insertPlainText(value)

    def goto_welcome_screen(self):  # Returns to welcome screen
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setFixedHeight(600)
        widget.setFixedWidth(1000)

    def goto_build_new_model_screen(self):  # Go to build_new_model_screen
        widget.setCurrentIndex(widget.currentIndex() + 1)
        widget.setFixedHeight(800)
        widget.setFixedWidth(1000)

    def fine_tuned_the_model_clicked(
            self):  # Action in case of fine-tuned the model button clicked, if all parameters were chosen as required it will start a new thread to fine-tune a language model
        self.start_loading_animation()
        language_model_name = self.LM_comboBox.currentText()
        task_model_name = self.task_comboBox.currentText() + ' model'
        if self.model_task_dict[language_model_name][task_model_name] == '':
            if self.task_data_set_file_name is None:
                msg_box_error = QMessageBox()
                msg_box_error.setIcon(QMessageBox.Critical)
                msg_box_error.setWindowTitle("error")
                msg_box_error.setText("You have to load task's dataset file")
                msg_box_error.buttonClicked.connect(self.stop_loading_animation)
                msg_box_error.exec_()
                return
        p = Thread(target=self.fine_tuned_the_model)
        p.start()

    def start_loading_animation(
            self):  # Starts the loading animation gif and disables needed widgets to prevent the user from interfering in the fine-tuning process.
        self.label_for_fine_tune_the_model_gif.setMovie(self.movie)
        self.movie.start()
        self.LM_comboBox.setEnabled(False)
        self.task_comboBox.setEnabled(False)
        self.choose_tasks_dataset_button.setEnabled(False)
        self.fine_tune_the_model_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.clear_page_button.setEnabled(False)

    def stop_loading_animation(
            self):  # Stops the loading animation gif and enabled disabled widgets while fine-tuning process.
        self.movie.stop()
        self.label_for_fine_tune_the_model_gif.clear()
        self.LM_comboBox.setEnabled(True)
        self.task_comboBox.setEnabled(True)
        self.choose_tasks_dataset_button.setEnabled(True)
        self.fine_tune_the_model_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.clear_page_button.setEnabled(True)

    # Fine-tuning the language model according to ULMFiT algorithm from stage 3, in case if there is already fine-tuned model, it will load the weights and will not preform fine-tuning
    def fine_tuned_the_model(self):
        model = Model(self.fine_tuning_text_update, self.fine_tuning_text_update)
        language_model_name = self.LM_comboBox.currentText()
        task_model_name = self.task_comboBox.currentText() + ' model'
        if self.model_task_dict[language_model_name][task_model_name] == '':
            saved_model_path = ROOT_DIR + self.model_task_dict[language_model_name]['base language model']
            self.current_model = \
                model.target_task_language_model_fine_tuning(saved_model_path, None, None, None, None, None, None)[0]
            sentences, sentiment, number_of_target_class_classes = dataset_preprocess.build_data_set_for_sentimental_analysis(
                self.task_data_set_file_name, self.model_task_dict[language_model_name]['language'], "text",
                "sentiment")

            x_train, x_test, y_train, y_test = train_test_split(sentences, sentiment, test_size=0.2,
                                                                shuffle=True)
            model_dir_name = ROOT_DIR + '\\savedModels\\' + self.LM_comboBox.currentText() + task_model_name.split()[0]
            model.fine_tuned_the_model(model_dir_name, number_of_target_class_classes, x_train, y_train, x_test, y_test)
            self.current_model = model.model
            final_task_model_dir = '\\savedModels\\' + self.LM_comboBox.currentText() + task_model_name.split()[0]
            self.model_task_dict[language_model_name][task_model_name] = final_task_model_dir
            self.app_shelve_db.update_model_task_dict(self.model_task_dict)  # update data base
            self.current_model.save(ROOT_DIR + final_task_model_dir)  # Save model
        else:
            saved_model_path = ROOT_DIR + self.model_task_dict[language_model_name][task_model_name]
            self.current_model = \
                model.target_task_language_model_fine_tuning(saved_model_path, None, None, None, None, None, None)[0]
        self.load_text_file_radioButton.setEnabled(True)
        self.insert_text_manually_radioButton.setEnabled(True)
        self.load_text_file_button.setEnabled(True)
        self.stop_loading_animation()

    def lm_combobox_activated(self):  # Add items to the task_comboBox according to chosen language model.
        self.task_comboBox.clear()
        nlp_tasks = self.model_task_dict[self.LM_comboBox.currentText()]['tasks']
        self.task_comboBox.addItems(nlp_tasks)

    def task_combobox_activated(self):
        if self.task_comboBox.currentText() == 'Authorship analysis':
            self.choose_tasks_dataset_button.setEnabled(False)
        else:
            self.choose_tasks_dataset_button.setEnabled(True)
        self.fine_tune_the_model_button.setEnabled(True)

    # action called by the radio button
    def change_load_text_status(self):
        # radio is checked
        if self.load_text_file_radioButton.isChecked():
            # making load_text_file_button Enable
            self.load_text_file_button.setEnabled(True)
            self.lineEdit.setEnabled(False)
        else:
            # making load_text_file_button disable
            self.load_text_file_button.setEnabled(False)
            self.lineEdit.setEnabled(True)

    def open_file_name_dialog(
            self):  # Opens QFileDialog and saves a path to the last file selected by the user on which the prediction will be performed
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Text files (*.txt)", options=options)
        if file_name:
            self.file_name = file_name

    def open_file_choose_task_data_set_name_dialog(
            self):  # Opens QFileDialog and saves a path to the last dataset selected by the user on which the language model will be fine tuned
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*)", options=options)
        if file_name:
            self.task_data_set_file_name = file_name

    def get_prediction_clicked(
            self):  # Action in case of get prediction button clicked, if all parameters were chosen as required it will start a new thread to predict an answer
        self.start_loading_animation()
        task_model_dataset = self.task_comboBox.currentText() + ' dataset'
        language_model_name = self.LM_comboBox.currentText()
        task_model_dataset_path = ROOT_DIR + self.model_task_dict[language_model_name][task_model_dataset]
        language_of_the_model = self.model_task_dict[language_model_name]['language']
        text_for_prediction = ''
        file_for_prediction = None
        if self.load_text_file_radioButton.isChecked():
            file_for_prediction = self.file_name
            self.file_name = None
        else:
            text_for_prediction = self.lineEdit.text().lower()
        if file_for_prediction is None and text_for_prediction == '':
            msg_box_error = QMessageBox()
            msg_box_error.setIcon(QMessageBox.Critical)
            msg_box_error.setWindowTitle("error")
            msg_box_error.setText("You have to insert text or load text file to get prediction")
            msg_box_error.buttonClicked.connect(self.stop_loading_animation)
            msg_box_error.exec_()
            return
        p = Thread(target=self.predict,
                   args=(task_model_dataset_path, language_of_the_model, file_for_prediction, text_for_prediction))
        p.start()

    def predict(self, task_model_dataset_path, language_of_the_model, file_for_prediction,
                text_for_prediction):  # Predict answer according to input and chosen model
        if self.task_comboBox.currentText() == 'Authorship analysis':
            if file_for_prediction is None:
                file_for_prediction = ROOT_DIR + "\\datasets\\user input text.txt"
                f = codecs.open(file_for_prediction, "w", 'utf-8')
                f.write(text_for_prediction)
                f.close()
            sequences, labels, total_unique_words, optional_sequences, optional_labels = dataset_preprocess.build_dataset_for_predicting_the_next_word(
                file_path=task_model_dataset_path, language=language_of_the_model,
                optional_file_path=file_for_prediction)
            train_predictions = self.current_model.predict(sequences)
            test_predictions = self.current_model.predict(optional_sequences)
            methods.get_scatter_plot(train_predictions, test_predictions)
            pixmap = QPixmap(ROOT_DIR + '\\outputs\\scatter_plot.jpg')
            self.place_for_prediction_plot_label.setPixmap(pixmap)
            self.place_for_prediction_plot_label.resize(pixmap.width(), pixmap.height())

        if self.task_comboBox.currentText() == 'Sentiment analysis':
            if text_for_prediction == '':
                f = codecs.open(file_for_prediction, "r", 'utf-8')
                text = f.read()
                f.close()
            else:
                text = self.lineEdit.text().lower()
            prediction = self.current_model.predict([text])
            answer = methods.get_sentiment_analysis_prediction(list(prediction[0]))
            self.place_for_prediction_plot_label.setText(answer)
        self.stop_loading_animation()


"""
BuildNewModelScreen - controller of the BuildNewModelScreen

parameters:
language_model_name: User's input name for the new language model
language: User's language choice for the language of the model 
tasks: User's tasks choice for the tasks that the model could preformed 
train_history: Dictionary saves the history of the model training process
accuracy_plot: Saves the path to plot.jpg file of the accuracy plot
loss_plot: Saves the path to plot.jpg file of the loss plot
model_dir_name_to_save: Path for the saved model if the user want to save it
current_model: saves an object of the current used model in BuildNewModelScreen
"""


class BuildNewModelScreen(QDialog):
    model_structure_text_update = pyqtSignal(str)
    epochs_text_update = pyqtSignal(str)

    def __init__(self):
        super(BuildNewModelScreen, self).__init__()
        self.language_model_name = None
        self.language = None
        self.tasks = None
        self.train_history = None
        self.accuracy_plot = None
        self.loss_plot = None
        self.model_dir_name_to_save = None
        self.current_model = None
        loadUi(ROOT_DIR + "/ui/buildNewModelScreen.ui", self)
        self.movie = QMovie(ROOT_DIR + "/img/loading.gif")
        self.return_to_main.clicked.connect(self.goto_main_screen)
        self.training_analysis_button.clicked.connect(self.goto_training_analysis_screen)
        self.load_general_corpus_dataset_button.clicked.connect(self.open_file_name_dialog)
        self.app_shelve_db = AppShelveDb()
        self.model_task_dict = self.app_shelve_db.get_model_task_dict()
        self.embedding_options_dict = self.app_shelve_db.get_embedding_options_dict()
        self.intermediate_layer_options = ["gru", "lstm"]
        self.choose_embeddings_comboBox.addItems(self.embedding_options_dict.keys())
        self.choose_embeddings_comboBox.activated.connect(self.choose_embeddings_comboBox_activated)
        self.choose_intermediate_layer_comboBox.addItems(self.intermediate_layer_options)
        self.general_corpus_dataset = None
        self.train_language_model_button.clicked.connect(self.train_language_model_button_clicked)
        self.training_text_cursor = self.training_text.textCursor()
        self.model_structure_text_update.connect(self.model_structure_text.setText)  # Add values
        # to QTextEdit-model_structure_text which present to
        # the user the  current model structure
        # (get values vro pyqtSignal from Model's object)
        self.epochs_text_update.connect(lambda value: self.change_epochs_text(value))
        self.save_model_button.clicked.connect(self.save_model_button_clicked)
        self.clear_page_button.clicked.connect(self.clear_page)

    def clear_page(self):  # Refreshes all parameters in the page and clear it
        self.language_model_name = None
        self.language = None
        self.tasks = None
        self.train_history = None
        self.accuracy_plot = None
        self.loss_plot = None
        self.model_dir_name_to_save = None
        self.current_model = None
        self.general_corpus_dataset = None
        self.save_model_button.setEnabled(False)
        self.training_analysis_button.setEnabled(False)
        self.training_text.clear()
        self.model_structure_text.clear()
        self.insert_model_name_line_edit.clear()
        self.choose_number_of_epochs_line_edit.clear()
        self.sentiment_checkBox.setChecked(False)
        self.autorship_checkBox.setChecked(False)

    def save_model_button_clicked(self):  # Action in case of save_model button clicked, if there is trained
        # language model to save (self.current_model is not None) it will start a thread fot save the language model
        self.start_loading_animation()
        if self.current_model is None:
            msg_box_error = QMessageBox()
            msg_box_error.setIcon(QMessageBox.Critical)
            msg_box_error.setWindowTitle("error")
            msg_box_error.setText("There is no model to save")
            msg_box_error.buttonClicked.connect(self.stop_loading_animation)
            msg_box_error.exec_()
            return
        p = Thread(target=self.save_model)
        p.start()

    def save_model(self):  # Saves the model in case of user clicks on save model button
        self.current_model.save(ROOT_DIR + self.model_dir_name_to_save)  # Save model
        # Build a dictionary for the new Language model and update the database
        model_dict = {'tasks': self.tasks,
                      'base language model': self.model_dir_name_to_save,
                      'language': self.language
                      }
        for task in model_dict['tasks']:
            model_dict[task + ' model'] = ''
            model_dict[task + ' dataset'] = ''
        # Example for model's dictionary
        # 'Russian with GRU sholo don': {'tasks': ['Authorship analysis'],
        #                                'base language model': "\\savedModels\\russianGRUNextWord",
        #                                'Authorship analysis model': "\\savedModels\\russianGRUNextWord",
        #                                'Authorship analysis dataset': "\\datasets\\And_Quiet_Flows_the_Don_1.txt",
        #                                'language': 'russian'
        #                                },
        self.model_task_dict[self.language_model_name] = model_dict
        self.app_shelve_db.update_model_task_dict(self.model_task_dict)
        self.stop_loading_animation()

    def change_epochs_text(self, value):  # Add values to QTextEdit-training_text which present to
        # the user the  training process on the execution of model training
        # (get values vro pyqtSignal from Model's object)
        self.training_text_cursor.movePosition(QTextCursor.End)
        self.training_text.insertPlainText(value)

    def goto_main_screen(self):  # go to MainScreen
        main_screen.refresh_model_task_dict()
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setFixedHeight(1000)
        widget.setFixedWidth(1300)

    def goto_training_analysis_screen(self):  # go to TrainingAnalysisScreen
        training_analysis_screen.set_loss_plot(self.loss_plot)  # set the path to plot.jpg file of the loss plot
        training_analysis_screen.set_accuracy_plot(
            self.accuracy_plot)  # set the path to plot.jpg file of the accuracy plot
        training_analysis_screen.set_plots()
        widget.setCurrentIndex(widget.currentIndex() + 1)
        widget.setFixedHeight(700)
        widget.setFixedWidth(1600)

    def open_file_name_dialog(self):  # Opens QFileDialog and saves a path to the general domain corpus dataset
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Text files (*.txt)", options=options)
        if file_name:
            self.general_corpus_dataset = file_name

    def choose_embeddings_comboBox_activated(
            self):  # Add languages to the choose_language_comboBox according to chosen embedding layer.
        self.choose_language_comboBox.clear()
        languages = self.embedding_options_dict[self.choose_embeddings_comboBox.currentText()]
        self.choose_language_comboBox.addItems(languages)

    def train_language_model_button_clicked(self):  # Action in case of train_language_model_button button
        # clicked, if all parameters were chosen as required it will start a new thread to train_language_model
        self.start_loading_animation()
        self.language_model_name = self.insert_model_name_line_edit.text()
        embeddings = self.choose_embeddings_comboBox.currentText()
        self.language = self.choose_language_comboBox.currentText()
        intermediate_layer = self.choose_intermediate_layer_comboBox.currentText()
        number_of_epochs = self.choose_number_of_epochs_line_edit.text()
        sentiment_analysis_task = self.sentiment_checkBox.isChecked()
        authorship_analysis_task = self.autorship_checkBox.isChecked()
        self.tasks = []
        if sentiment_analysis_task:
            self.tasks.append("Sentiment analysis")
        if authorship_analysis_task:
            self.tasks.append("Authorship analysis")
        for var in [self.language_model_name, embeddings, self.language, intermediate_layer, number_of_epochs,
                    self.general_corpus_dataset]:
            if var == '' or var is None or len(self.tasks) == 0:
                msg_box_error = QMessageBox()
                msg_box_error.setIcon(QMessageBox.Critical)
                msg_box_error.setWindowTitle("error")
                msg_box_error.setText("You have to fill all parameters for the new language model")
                msg_box_error.buttonClicked.connect(self.stop_loading_animation)
                msg_box_error.exec_()
                return
        if not number_of_epochs.isnumeric():
            msg_box_error = QMessageBox()
            msg_box_error.setIcon(QMessageBox.Critical)
            msg_box_error.setWindowTitle("error")
            msg_box_error.setText("The number of epochs must have a numeric value")
            msg_box_error.buttonClicked.connect(self.stop_loading_animation)
            msg_box_error.exec_()
            return
        self.accuracy_plot = ROOT_DIR + "\\outputs\\" + self.language_model_name + "_accuracy_plot.jpg"
        self.loss_plot = ROOT_DIR + "\\outputs\\" + self.language_model_name + "_loss_plot.jpg"
        self.model_dir_name_to_save = "\\savedModels\\" + self.language_model_name
        p = Thread(target=self.train_language_model,
                   args=(intermediate_layer, number_of_epochs, embeddings, self.language,))
        p.start()

    def train_language_model(self, intermediate_layer, number_of_epochs, embeddings,
                             language):  # Method  for train language model
        model = Model(self.epochs_text_update, self.model_structure_text_update)  # Init new instance of Model class
        sequences, labels, total_unique_words = dataset_preprocess.build_dataset_for_predicting_the_next_word(
            file_path=self.general_corpus_dataset, language=language)  # Builds dataset for language model training
        self.current_model, self.train_history = model.target_task_language_model_fine_tuning(model_dir_name=None,
                                                                                              intermediate_layer=intermediate_layer,
                                                                                              total_unique_words=total_unique_words,
                                                                                              sequences_train=sequences,
                                                                                              labels_train=labels,
                                                                                              needed_epochs=int(
                                                                                                  number_of_epochs),
                                                                                              embedding_name=embeddings)  # Train the new language model
        methods.get_loss_plot(self.train_history,
                              self.loss_plot)  # Generates loss plot and saves it in outputs directory
        methods.get_accuracy_plot(self.train_history,
                                  self.accuracy_plot)  # Generates accuracy plot and saves it in outputs directory
        self.save_model_button.setEnabled(True)
        self.training_analysis_button.setEnabled(True)
        self.stop_loading_animation()

    def start_loading_animation(self):  # Starts the loading animation gif and disables needed widgets to
        # prevent the user from interfering in the training process.
        self.label_for_train_language_model_gif.setMovie(self.movie)
        self.movie.start()
        self.insert_model_name_line_edit.setEnabled(False)
        self.choose_embeddings_comboBox.setEnabled(False)
        self.choose_language_comboBox.setEnabled(False)
        self.choose_intermediate_layer_comboBox.setEnabled(False)
        self.choose_number_of_epochs_line_edit.setEnabled(False)
        self.load_general_corpus_dataset_button.setEnabled(False)
        self.train_language_model_button.setEnabled(False)
        self.clear_page_button.setEnabled(False)

    def stop_loading_animation(
            self):  # Stops the loading animation gif and enabled disabled widgets while training process.
        self.movie.stop()
        self.insert_model_name_line_edit.setEnabled(True)
        self.choose_embeddings_comboBox.setEnabled(True)
        self.choose_language_comboBox.setEnabled(True)
        self.choose_intermediate_layer_comboBox.setEnabled(True)
        self.choose_number_of_epochs_line_edit.setEnabled(True)
        self.load_general_corpus_dataset_button.setEnabled(True)
        self.train_language_model_button.setEnabled(True)
        self.label_for_train_language_model_gif.clear()
        self.clear_page_button.setEnabled(True)


"""
TrainingAnalysisScreen - controller of the TrainingAnalysisScreen

parameters:
accuracy_plot: path to accuracy plot jpg
loss_plot: path to loss plot jpg
"""


class TrainingAnalysisScreen(QDialog):
    def __init__(self):
        super(TrainingAnalysisScreen, self).__init__()
        loadUi(ROOT_DIR + "/ui/trainingAnalysisScreen.ui", self)
        self.accuracy_plot = None
        self.loss_plot = None
        self.return_to_build_model.clicked.connect(self.goto_build_new_model_screen)

    def goto_build_new_model_screen(self):  # go to BuildNewModelScreen
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setFixedHeight(800)
        widget.setFixedWidth(1000)

    def set_loss_plot(self, plot_dir_name):  # setter for loss plot path
        self.loss_plot = plot_dir_name

    def set_accuracy_plot(self, plot_dir_name):  # setter for loss plot path
        self.accuracy_plot = plot_dir_name

    def set_plots(self):  # Show plots on the screen
        pixmap = QPixmap(self.accuracy_plot)
        self.place_for_accuracy_plot.setPixmap(pixmap)
        self.place_for_accuracy_plot.resize(pixmap.width(), pixmap.height())
        pixmap = QPixmap(self.loss_plot)
        self.place_for_loss_plot.setPixmap(pixmap)
        self.place_for_loss_plot.resize(pixmap.width(), pixmap.height())


# main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    welcome = WelcomeScreen()
    widget.addWidget(welcome)
    main_screen = MainScreen()
    widget.addWidget(main_screen)
    new_model_screen = BuildNewModelScreen()
    widget.addWidget(new_model_screen)
    training_analysis_screen = TrainingAnalysisScreen()
    widget.addWidget(training_analysis_screen)
    widget.setFixedHeight(600)
    widget.setFixedWidth(1000)
    ag = QDesktopWidget().availableGeometry()
    sg = QDesktopWidget().screenGeometry()
    x = (ag.width() - widget.width()) // 2
    y = (ag.height() - widget.height()) // 3
    widget.move(x, y)
    widget.setWindowTitle('ULMFiT research tool')
    widget.show()
    methods = MiscMethods()
    dataset_preprocess = DatasetPreprocess()
    try:
        sys.exit(app.exec())
    except:
        print("exiting")
