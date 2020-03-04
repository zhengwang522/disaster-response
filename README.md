# Disaster Response Pipeline Project

### Summary

This project is to analyze disaster data gathered from Figure Eight https://www.figure-eight.com/ to build a model for an API that classifies disaster messages. The training data contain about 26000 samples. The classification is for multi-label, each new message will be classified into one or more of the 36 disaster categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py sqlite:///data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
