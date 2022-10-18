## Project Summary
This project is sponsored by Figure Eight with the goal to build a ETL & ML pipeline to classify emergency messages and establish a web app where users such as (emergency worker) can input a new message and get classification results in several categories.

## File Descriptions

### app  
    ├── run.py                           # Flask file that runs app
    └── templates   
        ├── go.html                      # Classification result page of web app
        └── master.html                  # Main page of web app    


### data                   
    ├── disaster_categories.csv          # Dataset including all the categories  
    ├── disaster_messages.csv            # Dataset including all the messages
    └── process_data.py                  # Data cleaning

### models
    └── train_classifier.py              # Train ML model           

### README.md


## Installation Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
Note:  Skip the above and run the following steps directly if DisasterResponse.db and claasifier.pkl already exist.

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/

## Results

### Landing Page 
<img width="1888" alt="Screen Shot 2022-10-18 at 3 29 38 AM" src="https://user-images.githubusercontent.com/23645903/196406792-fc7445fb-00d0-49be-b660-ab5cde35c7b7.png">
### Sample Output
<img width="1888" alt="Screen Shot 2022-10-18 at 3 16 56 PM" src="https://user-images.githubusercontent.com/23645903/196555546-b568b18f-b8ec-4da2-80a7-f9f84b28ff8a.png">


https://052b8c06e71c40e99045311a7e35ff96-3000.udacity-student-workspaces.com/
