# world-bank-climate-regression
Code to create an NLP regression to predict the percentage of project finance that relates to climate

## Installation
```
python3 -m virtualenv venv
source venv/bin/activate
```

If you want to make your own copy of the data and train a new model, copy .env-example to .env and fill out your Huggingface token.

## Data collection and upload
```
python3 code/wb_api_climate.py
python3 code/upload_climate_dataset.py
```

## Training a new model
```
python3 code/train_wb_regression_model.py
```

## Inference model
```
python3 code/inference_wb_regression_model.py
```