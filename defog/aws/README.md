# Deploying to AWS Lambda
This folder contains the code required to deploy defog as a serverless function on AWS Lambda.

## Setup
If you are using the CLI, please run `defog init` and follow the prompts, after which you can just initialize the class with `defog = Defog()`. We will read the connection settings from the `~/.defog/connection.json` file it creates.
If you prefer to set the API key and DB credentials manually, you can pass it to the `Defog` class in the `app.py` file. 

## Deploying
To deploy the function, you can run `defog deploy aws` from the CLI. This will use chalice under the hood to create a new Lambda function and API Gateway endpoint. You can then use the URL provided by the CLI to make requests to the API. 

Should you desire to customize the individual key-value settings in the chalice config like the `app_name`, `version`, you can do so by passing them as cli arguments like so:
```bash
defog deploy aws --app_name defog --version 2.0
```
The full list of configurable options can be found [here](https://chalice-fei.readthedocs.io/en/latest/topics/configfile.html#lambda-specific-configuration). Note that we do not pass in nested variables from the cli currently. Please feel free to reach out if you need to customize your deployment with nested variables.

## How it works
We start by creating a folder in `~/.defog/aws`, which will host all of the files required (`app.py`, etc). Note that chalice requires a subfolder named `.chalice` with a `config.json` file in it to specify its configuration. We provide a basic template in our library (similar to what `chalice new-project ...` would create), which we will read from, add the necessary environment variables for authorizing the DB connection, and save the final results in the `~/.defog/aws/.chalice/config.json` file. Note that we will be serializing the api key, db type and db creds into json first, and then base64 encode it, finally passing it as `DEFOG_CREDS_64` to the Lambda function. This is primarily because of the difficulty of passing nested values around the chalice api. Upon a successful deployment, you should see a new folder named `~/.defog/aws/.chalice/deployed` with the deployment artifacts. You can also run `chalice deploy` directly from the `~/.defog/aws` folder to redeploy the function, should you update it, though we would discourage you from doing so unless you know what you are doing (aka familiar with `chalice`). so you can make changes to the code in this folder directly, like the `app.py` file, which contains the serverless function. 