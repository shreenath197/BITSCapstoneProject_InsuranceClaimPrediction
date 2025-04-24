This project aims to build a machine learning model that predicts whether a policyholder will file an insurance claim in the next year. The model is trained and tested using historical customer data to help insurance companies manage risk and make informed decisions.

Project Structure -

1. Run Insurance_Claim_Prediction_GRP-13.ipynb: Main Jupyter Notebook containing data exploration, preprocessing, model training, evaluation, and predictions.
   Input DATA : INPUT_DATASET > insurance_dataset_corr.csv

2. FASTAPI File : claim_pred_model.py [Data Validation], main_all.py [Main fastapi program]
     Input DATA : FAST_API_INPUT_CSV > claim_sample_data.csv

   	   Execution Steps in Local machine: 
           Step1: Run the Project file (.ipynb)
           Step2: Open Command prompt/ anaconda prompt terminal.
	   Step3: Change directory to the folder containing all the programs like Project file(.ipynb) , Class file (.py on BASEMODEL), main file(.py) [Local machine folder path]
                  Command: cd C:\Users\Abhijeet\AIML\BITS_Capstone_Project\NEW\CLAIM_FASTAPI
           Step4: To execute FASTAPI application execute uvicorn command
                   Prerequisite: pip install uvicorn (if not installed already)
				 pip install fastapi (if not installed already)
				 pip install python-multipart (if not installed already)
				 Command to execute: uvicorn main_all:app –reload
				 [main_all - main file name, app - instance of fastapi in the main.py file]

	   Step5: Post Application startup complete, copy the highlighted http URL and paste it in web browser to access FastAPI swagger UI
           Step6: User can upload the csv input file (claim_sample_data.csv) and click on execute to get the prediction done for target attribute claim amount.
           Step7: Validate JSON which has claim prediction amount

