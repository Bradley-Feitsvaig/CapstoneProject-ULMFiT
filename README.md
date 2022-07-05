1. General Description 

The ULMFiT research tool main goal is to examine the operation of the ULMFiT 
algorithm. With the help of the tool, the algorithm can be applied to different languages and 
NLP tasks. For the different NLP tasks that are available in the system, the algorithm is 
executed according to the need of the specific task, thus all the algorithm stages can be 
examined separately.
There are two main stages in which the algorithm can be executed – the training of a general 
domain language model and the fine tuning of the language model for a specific NLP task. A 
different window is associated with each stage of the algorithm.
The main window is for fine tuning of an existing language model. The second window (build 
new model window) is for the purpose of creating a custom-made language model based on 
the ULMFiT principles for further research.
The tool is oriented towards researchers who want to examine further the ULMFiT 
application on different models and languages.

2. System Operation

Building a new language model window – on the "building new language model" 
window, a model name must be inserted which will be used when saving the model. A word 
embedding layer is chosen for the model, together with the model language (English or 
Russian), the intermediate layers (GRU or LSTM) and the number of epochs for which the 
model will be trained. A general corpus dataset in the form of a txt file needed to be loaded and 
the target task is chosen whether its Authorship analysis or Sentiment analysis. In the end of 
Braude College of Engineering 2022
Page 20 of 22
the model creation process, the "Training analysis" button will be enabled and will lead to a 
separate window which will show available graphs of the training process.
Training analysis window– In this window, the graphs of the training process will be 
shown.
Main window – The first section of the main window is model fine tuning. Choose a 
model from the list of the available trained language models and a desired NLP task from the 
list of available tasks. In case of sentimental analysis, load a specific target task dataset, it must 
be in the form of a CSV file (an existing CSV file is located inside the datasets directory named 
"Twitter US Airlines Sentiment dataset.csv") – the header for the tweets should be named "text" 
and for the label(sentiment) the header should be named "sentiment". The sentiments should be 
one of the following: "positive", "neutral" or "negative". In the case of authorship analysis, the 
language model is already satisfying the requirements of the specific task, thus, it is not 
necessary to proceed with the next stage of the ULMFiT algorithm and the "Choose task's 
dataset" button will be disabled. In case a model which is already fine-tuned is selected, the 
"Fine tune model" button will only load an already tuned model and the fine tuning will not 
execute. To fine tune the model, click on "Fine tune the model" button. In the fine-tuning 
process, the model architecture and tuning progress is shown on the lower-right section of the 
window.
The second section of the main window is the execution of the NLP task. After fine-tuning the 
language model, there are two possibilities: (1) load a text file with the "Load" button in the 
format of a ".txt". The file should have the content of a book for the authorship analysis task or 
a sentence for the sentiment analysis task. (2) Insert text manually with the "Insert text 
manually" bar. After the text is loaded, click on the "Predict" button. The system will show the 
prediction results on the lower-left side of the window.

3. Environment Specification

Using the tool requires configuration to the environment. To run the application, 
please go through the following steps:
1) Confirm that the latest Anaconda version is installed, and the latest Python 
environment (configured with Anaconda) is available.
2) Create a new environment using Anaconda.
3) Run "install.bat" which is attached to the project directory. In case of a failure 
during installation, install manually the following libraries:
• pyqt5
• sklearn
• pandas
• nltk
• keras
• tensorflow
• matplotlib
• tensorflow_hub
• tensorflow_text
*For the nltk library, an additional download is required:
On command prompt type –
Python
>>import nltk
>>nltk.download('stopwords')
4) Navigate using command prompt to the project directory and run the "app.py" 
file: python app.py


*Pay attention that the loaded files should be in the same language all the way through the
process according to the language of the model.

