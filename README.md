# Disaster Response Management System
## Part of the Data Scientist Nanodegree Program by Udacity

### Overview
This project is a component of the [Data Scientist Nanodegree Program at Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025). It involves developing a machine learning pipeline to classify messages originating from real-world disaster situations. The goal is to route these categorized messages to the appropriate disaster response agencies effectively.

### Prerequisites
To execute this project, make sure your system has the following packages installed:
- Python 3.7
- Pandas
- scikit-learn
- Numpy
- SQLalchemy
- NLTK
- Punkt
- Flask
- Plotly

#### Installation Requirements for Local Deployment
To operate the web dashboard on your local machine, it's essential to install the requisite Python packages. These dependencies are listed in the `requirements.txt` file located in the project's repository. Install them by running the following command:

```bash
pip install -r requirements.txt
```

### Project Architecture
The project is organized into several key components:
- `ETL Pipeline Preparation.ipynb`: Jupyter notebook containing code for setting up the ETL pipeline.
- `ML Pipeline Preparation.ipynb`: Jupyter notebook with code to prepare the machine learning pipeline.
- `models/train_classifier.py`: Python script for training the classifier and saving the model.
- `data/process_data.py`: Python script for data cleansing and ETL pipeline storage.
- `data/DisasterResponse.db`: Database containing processed messages and their categories.
- `data/disaster_categories.csv`: CSV file listing message categories.
- `data/disaster_messages.csv`: CSV file holding the disaster messages.
- `app/run.py`: Flask file to launch the web application.
- `app/templates/`: Folder containing web application templates, including `master.html` and `go.html`.

### How to Run the Project
1. Navigate to the project root directory and execute the following commands to set up the database and machine learning model:
    - For the ETL pipeline:  
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - For the ML pipeline:  
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To launch the web application, navigate to the app directory and run:  
   `python run.py`

### Licensing
- The dataset employed for this project is courtesy of Figure Eight.
- This project is developed as part of the [Udacity](https://www.udacity.com) Data Scientist Nanodegree Program.
