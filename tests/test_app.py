import sys
from unittest import TestCase

from app import MainScreen, WelcomeScreen, BuildNewModelScreen, TrainingAnalysisScreen
from PyQt5.QtWidgets import QDialog, QStackedWidget, QApplication

from shelve_db import AppShelveDb


class TestMainScreen(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.widget = QStackedWidget()
        cls.main_screen = MainScreen()

    """
    checks that the window is returned to the default state after pressing the clear button.
    """
    def test_refresh_page(self):
        self.main_screen.refresh_page()
        self.assertIsNone(self.main_screen.file_name)
        self.assertIsNone(self.main_screen.task_data_set_file_name)
        self.assertIsNone(self.main_screen.current_model)
        self.assertFalse(self.main_screen.fine_tune_the_model_button.isEnabled())
        self.assertFalse(self.main_screen.load_text_file_button.isEnabled())
        self.assertFalse(self.main_screen.lineEdit.isEnabled())
        self.assertFalse(self.main_screen.predict_button.isEnabled())
        self.assertFalse(self.main_screen.insert_text_manually_radioButton.isEnabled())
        self.assertFalse(self.main_screen.load_text_file_radioButton.isEnabled())
        self.assertFalse(self.main_screen.insert_text_manually_radioButton.isChecked())
        self.assertEqual(self.main_screen.lineEdit.text(), '')
    """
    checks if the LM_comboBox displays all the available language models in the system.
    """
    def test_refresh_model_task_dict(self):
        self.main_screen.refresh_model_task_dict()
        app_shelve_db = AppShelveDb()
        model_task_dict = app_shelve_db.get_model_task_dict()
        AllItems = [self.main_screen.LM_comboBox.itemText(i) for i in range(self.main_screen.LM_comboBox.count())]
        self.assertEqual(list(model_task_dict.keys()), AllItems)

    """
    checks if at the start of a GUI operation, the LM_comboBox,task_comboBox, choose_tasks_dataset_button, 
    fine_tune_the_model_button, predict_button and clear_page_button widgets are disabled.
    """
    def test_start_loading_animation(self):
        self.main_screen.start_loading_animation()
        self.assertFalse(self.main_screen.LM_comboBox.isEnabled())
        self.assertFalse(self.main_screen.task_comboBox.isEnabled())
        self.assertFalse(self.main_screen.choose_tasks_dataset_button.isEnabled())
        self.assertFalse(self.main_screen.fine_tune_the_model_button.isEnabled())
        self.assertFalse(self.main_screen.predict_button.isEnabled())
        self.assertFalse(self.main_screen.clear_page_button.isEnabled())

    """
     checks if at the end of a GUI operation, the LM_comboBox,task_comboBox, choose_tasks_dataset_button, 
     fine_tune_the_model_button, predict_button and clear_page_button widgets are enabled.
    """
    def test_stop_loading_animation(self):
        self.main_screen.stop_loading_animation()
        self.assertTrue(self.main_screen.LM_comboBox.isEnabled())
        self.assertTrue(self.main_screen.task_comboBox.isEnabled())
        self.assertTrue(self.main_screen.choose_tasks_dataset_button.isEnabled())
        self.assertTrue(self.main_screen.fine_tune_the_model_button.isEnabled())
        self.assertTrue(self.main_screen.predict_button.isEnabled())
        self.assertTrue(self.main_screen.clear_page_button.isEnabled())

    """
    checks if the model task combo box is displaying only the relevant tasks according to the selected model.
    """
    def test_task_combobox_activated(self):
        self.main_screen.lm_combobox_activated()
        app_shelve_db = AppShelveDb()
        model_task_dict = app_shelve_db.get_model_task_dict()
        currentText = self.main_screen.LM_comboBox.currentText()
        nlp_tasks = model_task_dict[currentText]['tasks']
        AllItems = [self.main_screen.task_comboBox.itemText(i) for i in range(self.main_screen.task_comboBox.count())]
        self.assertEqual(nlp_tasks, AllItems)


class TestBuildNewModelScreen(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.widget = QStackedWidget()
        cls.new_model_screen = BuildNewModelScreen()

    def test_clear_page(self):
        self.new_model_screen.clear_page()
        self.assertIsNone(self.new_model_screen.language_model_name)
        self.assertIsNone(self.new_model_screen.language)
        self.assertIsNone(self.new_model_screen.tasks)
        self.assertIsNone(self.new_model_screen.train_history)
        self.assertIsNone(self.new_model_screen.accuracy_plot)
        self.assertIsNone(self.new_model_screen.loss_plot)
        self.assertIsNone(self.new_model_screen.model_dir_name_to_save)
        self.assertIsNone(self.new_model_screen.current_model)
        self.assertIsNone(self.new_model_screen.general_corpus_dataset)
        self.assertFalse(self.new_model_screen.save_model_button.isEnabled())
        self.assertFalse(self.new_model_screen.training_analysis_button.isEnabled())
        self.assertFalse(self.new_model_screen.sentiment_checkBox.isChecked())
        self.assertFalse(self.new_model_screen.autorship_checkBox.isChecked())
        self.assertEqual(self.new_model_screen.insert_model_name_line_edit.text(), '')
        self.assertEqual(self.new_model_screen.choose_number_of_epochs_line_edit.text(), '')

    """
    checks if the choose language combo box is displaying only the relevant languages 
    according to the selected embedding.
    """
    def test_choose_embeddings_combo_box_activated(self):
        self.new_model_screen.choose_embeddings_comboBox_activated()
        app_shelve_db = AppShelveDb()
        embedding_options_dict = app_shelve_db.get_embedding_options_dict()
        current_text = self.new_model_screen.choose_embeddings_comboBox.currentText()
        languages = embedding_options_dict[current_text]
        AllItems = [self.new_model_screen.choose_language_comboBox.itemText(i) for i in range(self.new_model_screen.choose_language_comboBox.count())]
        self.assertEqual(languages, AllItems)

    def test_start_loading_animation(self):
        self.new_model_screen.start_loading_animation()
        self.assertFalse(self.new_model_screen.insert_model_name_line_edit.isEnabled())
        self.assertFalse(self.new_model_screen.choose_embeddings_comboBox.isEnabled())
        self.assertFalse(self.new_model_screen.choose_language_comboBox.isEnabled())
        self.assertFalse(self.new_model_screen.choose_intermediate_layer_comboBox.isEnabled())
        self.assertFalse(self.new_model_screen.choose_number_of_epochs_line_edit.isEnabled())
        self.assertFalse(self.new_model_screen.load_general_corpus_dataset_button.isEnabled())
        self.assertFalse(self.new_model_screen.train_language_model_button.isEnabled())
        self.assertFalse(self.new_model_screen.clear_page_button.isEnabled())


    def test_stop_loading_animation(self):
        self.new_model_screen.stop_loading_animation()
        self.assertTrue(self.new_model_screen.insert_model_name_line_edit.isEnabled())
        self.assertTrue(self.new_model_screen.choose_embeddings_comboBox.isEnabled())
        self.assertTrue(self.new_model_screen.choose_language_comboBox.isEnabled())
        self.assertTrue(self.new_model_screen.choose_intermediate_layer_comboBox.isEnabled())
        self.assertTrue(self.new_model_screen.choose_number_of_epochs_line_edit.isEnabled())
        self.assertTrue(self.new_model_screen.load_general_corpus_dataset_button.isEnabled())
        self.assertTrue(self.new_model_screen.train_language_model_button.isEnabled())
        self.assertTrue(self.new_model_screen.clear_page_button.isEnabled())

