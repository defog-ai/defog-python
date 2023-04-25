# Deploying to Google Cloud Functions
This folder contains the code required to deploy defog as a serverless function on Google Cloud Run.

## Setup
If you are using the CLI, you can just run `defog init` and follow the prompts, after which you can just initialize the class with just `defog = Defog()`. We will read the connection settings from the `~/.defog/connection.json` file.
If you prefer to set the API key and DB credentials manually, you can pass it to the `Defog` class in the `main.py` file. 

## Deploying
The easiest way to deploy this function is to use the defog CLI:
```
defog deploy gcp
```
Note that you will need to have the `gcloud` CLI installed and configured to deploy to your project. You can find more information on how to do so [here](https://cloud.google.com/sdk/docs/install). 

If you would like to deploy the function manually after modifying `main.py` or the deployment configuration below, you can run the following command from this folder:

```
gcloud functions deploy test-defog \
--gen2 \
--runtime=python310 \
--region=us-central1 \
--source=. \
--entry-point=defog_query_http \
--trigger-http \
--allow-unauthenticated \
--max-instances=1
```