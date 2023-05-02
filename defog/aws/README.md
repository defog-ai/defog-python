# Deploying to AWS Lambda
This folder contains the code required to deploy defog as a serverless function on AWS Lambda.

## Setup
If you are using the CLI, you can just run `defog init` and follow the prompts, after which you can just initialize the class with just `defog = Defog()`. We will read the connection settings from the `~/.defog/connection.json` file.
If you prefer to set the API key and DB credentials manually, you can pass it to the `Defog` class in the `main.py` file. 

## Deploying
To deploy the function, you can run `defog deploy aws` from the CLI. This will use chalice under the hood to create a new Lambda function and API Gateway endpoint. You can then use the URL provided by the CLI to make requests to the API. Note that we deploy the `defog/aws` folder as is, so you can make changes to the code in this folder directly. Note that chalice requires a folder named `.chalice` with a `config.json` file in it. We provide a basic template `defog/aws/.chalice/base_config.json`, which we will read from, add the necessary environment variables, and save the final results in the `defog/aws/.chalice/config.json` file. Note that we will be serializing all of the environment variables into json first, and then base64 encode it, finally passing it as `DEFOG_CREDS_64` to the Lambda function. This is primarily because of the difficulty of passing nested values around the chalice api.