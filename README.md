# H1B and PERM prediction

An app to predict CONFIRM/DENIED for the H1B and PERM application.

### To run the app

1. create virtual environment
virtualenv -p python3.7 venv
2. install packages
pip install -r conf/requirements.txt
3. spin on app
python ui.py

#### Note

In the directory dialog, the path should be a folder, e.g. /user/.
* the training data will be kept in /user/input
* the prediction data will be kept in /user/predict
* the code will be in /user/h1permprediction
* the model will be in /user/model
* the temprory files will be in /user/temp
* the output, model and temporary folder will be generated during execution under base diretory.