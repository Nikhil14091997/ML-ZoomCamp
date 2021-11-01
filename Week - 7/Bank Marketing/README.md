<h1> Problem description </h1
<p> 
  How to Optimize marketing campaigns using data from previous marketing campaign.
  The classification goal is to predict if the client will subscribe a term deposit (variable y).
  Different classification algorithms will be used for evaluation purpose i.e.
  <ul>
    <li>Logistic Regression</li>
    <li>Logistic Regression with hyperparameter tuning</li>
    <li>Random Forest Classification</li>
    <li>Gradient Boosting</li>
    <li>SVM</li>
    <li> XGBoost </li>
</ul>
  
  
  1. <h3>Title: Bank Marketing </h3>

  2. Sources
   Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012

  3. Past Usage:

  The full dataset was described and analyzed in:

  S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, 
  Portugal, October, 2011. EUROSIS.
  
  This dataset is public available for research. The details are described in [Moro et al., 2011]. 
  Please include this citation if you plan to use this database:

  [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

  Available at: [pdf] http://hdl.handle.net/1822/14838
  
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

  
<h1> EDA - Exploratory Data Analysis </h1>
  <p> In depth exploratory data analysis has been performed to
  <ol>
    <li> Eliminate Unnecessary attributes </li>
    <li> Identifying most important attributes </li>
    <li> Calculating mutual correlation </li>
    <li> Different visualization graphs to discern deeper understanding of data </li>
  </ol>
  </p>
 
 <h1> Model training </h1>
 <p> Trained multiple models and tuned their parameters</p>
 
 <h1> Exporting notebook to script </h1>
 <p> The logic for training the model is exported to a separate script </p>
 
 <h1> Model deployment </h1>
 <p> Model is deployed with Flask.
    <ul>
      <li>In order to predict we need to first load the previous saved model and use a prediction function in a special route.</li>
      <li>To load the previous saved model we use the code below:</li>
    </ul>
          <p> 
            
            import pickle
            model_file = 'final_model=1.0.bin'

            # loading the dict-vectorizer and model
            with open(model_file, 'rb') as f_in:
                dv, model = pickle.load(f_in)
            print("DictVectoriser and Model are loaded.")
  </p>
      
</p>
 
 <h1> Dependency and enviroment management </h1>
 <p> Provided a file with dependencies and used virtual environment(pipenv) . Name of virtual environment is bank-marketing </p>
 
 <h1> 	Containerization </h1>
 To run docker without `sudo`, follow [this instruction](https://docs.docker.com/engine/install/linux-postinstall/).
  - Once our project was packed in a Docker container, we're able to run our project on any machine.
  - First we have to make a Docker image. In Docker image file there are settings and dependecies we have in our project. To find Docker images that you need you can simply search the [Docker](https://hub.docker.com/search?type=image) website.
  - Here a Docker file is written we'll explain it below.(There should be no comments in Docker file, So remove the comments if you want to copy it)
  - '''
    FROM python:3.8.10-slim

    RUN pip install pipenv

    WORKDIR /app

    COPY ["Pipfile", "Pipfile.lock", "./"]

    RUN pipenv install --system --deploy

    COPY ["predict.py", "final_model=1.0.bin", "./"]

    EXPOSE 9697

    ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9697", "predict:app"]
  '''
  If we don't put the last line ENTRYPOINT, we will be in a python shell. Note that in Docker we made put in double quotes, This is because of the spaces. We have to ignore   spaces in a command and put the characters in double quotes.(See ENTRYPOINT for example)
   - After creating the Dockerfile and writing the settings we want in it, We need to build it with the command below.
   - ```
      docker build -t bank-marketing .
     ```
      With _-t_ command We're specifying the name churn-prediction for this Dockerfile.
   - To run it, Simply execute the command below:
   - ```
      docker run -it -p 9697:9697 bank-marketing
     ```
     Here we use the option _-it_ in order to the Docker run from terminal and shows the result. 
     The _-p_ parameter is used to map the 9697 port of the Docker to 9697 port of our machine.(first 9696 is the Docker container port and the last is port number of our  machine)
   - At last you've deployed your prediction app inside a Docker continer.
