# study-predictor

This repository serves as a collection of backend implementations of certain models, e.g. BERT with tensorflow, 
GPT with huggingface, and many more.

The backend relies on FastAPI (https://fastapi.tiangolo.com/).

## Start developing:

1. Create a new branch for each model. The naming convention is as follows: `mlBackend_modelName_optionalFunctionName`, e.g. `huggingface_GPT2`, 
   `tensorflow_BERT`, or `huggingface_T5_summarization`
2. Inside the directory `code/app` create a `.env` file. Add content as shown below.
   ```
   PUBLIC_PREDICTION = False
   API_KEY = "YOUR-API-KEY"
   ```
   If `PUBLIC-PREDICTION` is set to True, authentication will be bypassed. If set to False, you need to send the `API_KEY` 
   with every HTTP request as `X-API-Key` header in order to get authenticated.
3. The minimum requirement is to offer a `predict` as well as a `ping` endpoint in `code/app/api/v1/machine_learning.py`

## Start service:

1. cd into root directory
2. Run `docker-compose up`
3. The API will be available at `http://localhost:8008/` a documentation can be found at `http://localhost:8008/docs`

## After modifying requirements.txt:

1. rebuild docker container `docker-compose build`

## WIP: Tensorflow example

This tensorflow example is based on https://www.tensorflow.org/tutorials/text/classify_text_with_bert

In particular, we use a small-BERT to run a sentiment analysis on IMDB movie reviews. The goal is to infer the rating
from the movie-review text.

IMDB movie reviews: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
small-Bert model used in this example: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1
small-Bert pre-processor: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1

## Directories

Directories are:

- `code/app/datasets` holds the IMDB movie-reviews 
- `code/app/model` holds the small-BERT model
- `code/app/preprocess` holds the BERT pre-processing

Important: Download the files listed above and extract them to the directories