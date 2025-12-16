End-to-End MLOps Pipeline for Loan Eligibility Prediction


This project involves building an MLOps pipeline for a Loan Eligibility Prediction model using Python, deployed on Google Cloud Platform (GCP). The pipeline was designed to automate the entire process of model development, deployment, and monitoring using various GCP services, including Cloud Build, Cloud Run, Cloud Source Repository, Docker, and Flask. This setup ensures scalable, automated, and efficient machine learning operations.

Key Components:
Model Development:

A machine learning model was developed to predict loan eligibility based on various applicant features (e.g., income, credit score, loan amount). Python libraries like pandas, scikit-learn, and XGBoost were used to build and train the model.
Flask API:

A Flask API was created to serve the trained model. This API accepts user input (applicant details) via HTTP requests, runs the prediction through the model, and returns the loan eligibility result.
Docker Containerization:

The Flask application, along with all its dependencies, was containerized using Docker. This ensures that the application runs consistently across different environments, making it easier to deploy and scale.
Source Code Management:

The project’s source code was managed using Cloud Source Repository and Git, allowing for version control and team collaboration. The integration with GCP ensures seamless deployment and continuous integration (CI).
Automated Deployment with Cloud Build:

Google Cloud Build was used to automate the entire deployment pipeline. Each time a new update or change is pushed to the repository, Cloud Build automatically triggers the deployment process, ensuring that the latest version of the model is always available.
Model Deployment with Cloud Run:

The containerized Flask API was deployed using Cloud Run, a fully managed platform for deploying containerized applications. Cloud Run ensures that the API is scalable, highly available, and automatically handles incoming requests without the need for manual intervention.
Scalable and Automated Operations:

The pipeline leverages GCP’s services for efficient model deployment, version control, and CI/CD. This setup enables automatic updates, scalability, and monitoring, making it an ideal solution for production-level ML operations.
How the Pipeline Works:
Data Preprocessing:

The dataset (e.g., CSV file) containing applicant information is preprocessed using pandas. This includes handling missing values, encoding categorical features, and scaling numerical features.
Model Training:

The machine learning model is trained using scikit-learn or XGBoost, depending on the complexity and performance requirements. Hyperparameter tuning and cross-validation are performed to improve the model’s accuracy.
Flask API:

After training the model, a Flask-based API is created to expose the model's prediction functionality. The API receives input from the user, processes it, and returns the loan eligibility prediction.
Dockerizing the Application:

The entire Flask application, including the trained model and necessary dependencies, is containerized using Docker. A Dockerfile is created to specify the build instructions for the container.
Cloud Source Repository & Git:

The source code for the project is stored and versioned in Cloud Source Repository. Git is used for version control, and any changes pushed to the repository trigger the deployment pipeline.
Automated Deployment with Cloud Build:

Cloud Build automates the build, test, and deployment processes. Upon changes in the source repository, Cloud Build automatically builds the Docker image, runs tests, and deploys the application to Cloud Run.
Deploying with Cloud Run:

The containerized application is deployed to Cloud Run, where it can scale based on incoming traffic. Cloud Run automatically handles traffic distribution and server management.
Steps to Set Up the MLOps Pipeline:
Set Up Google Cloud Platform (GCP):

Create a GCP account and set up a new project.
Enable the necessary APIs: Cloud Build, Cloud Run, Cloud Source Repositories.
Clone the Repository:

Clone the repository to your local environment or GCP Cloud Shell.
Install Required Python Libraries:

Install the necessary Python libraries for model development:
bash
Copy code
pip install pandas scikit-learn xgboost flask
Train the Model:

Train the loan eligibility prediction model using the dataset, applying feature engineering and model evaluation techniques.
Create a Flask API:

Develop a Flask API to serve the trained model. The API should include a route that accepts POST requests with applicant data and returns the eligibility prediction.
Dockerize the Flask Application:

Create a Dockerfile to containerize the Flask application. Ensure the necessary dependencies are included.
Push Code to Cloud Source Repository:

Push the source code to Cloud Source Repository for version control and CI/CD integration.
Set Up Cloud Build:

Create a cloudbuild.yaml file to configure the build and deployment process with Cloud Build.
Deploy to Cloud Run:

Deploy the Dockerized Flask API to Cloud Run for scalable and automated serving of predictions.
Test the API:

Once deployed, you can test the API by sending POST requests with applicant data, and it will return the loan eligibility result.

For any questions or suggestions, feel free to reach out:
Raj Purohith Arjun
Email: raj2001@tamu.edu
LinkedIn: https://www.linkedin.com/in/raj-purohith-arjun-20a652200/
