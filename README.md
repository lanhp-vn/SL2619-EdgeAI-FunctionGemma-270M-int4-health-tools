
# Deployment instructions

## Introduction

Once your model has successfully been fine-tuned, it is ready for deployment.

This page outline two ways to deploy your model. Which method you choose depends on which approach fits best with your other applications.

Note that the following instructions have been tested on a cloud VM with an NVIDIA GPU (i.e. AWS EC2 g6e.xlarge).

## Setup

Extract the files from the model tarball in the same directory you are planning to work in. You should see the following files:
```
├── model/
├── model-adapters/
├── model_client.py
├── README.md
```

To get started, we recommend setting up a fresh environment.

Run the following to create and activate a virtual environment:
```bash
 python -m venv serve
 source serve/bin/activate
```


## Option 1: Ollama

Install Ollama by following the instructions outlined here:
https://ollama.com/

Install OpenAI:
```bash
pip install openai
```

Change into the `model/` directory, you should see the following files:
```
├── Modelfile
├── config.json
├── model.safetensors
├── ...
```

Create the model:
```bash
ollama create model -f Modelfile
```

You can then use the python script (`model_client.py`) to invoke the model, which allows you to pass in your own question and context:
```bash
python model_client.py --question "QUESTION" --context "CONTEXT"
```

If you invoke this script without providing the `--question` and `--context` arguments, an example from your test data will be used so you can familarize yourself with the output.


## Option 2: vLLM

Install vLLM and OpenAI using pip:
```bash
pip install vllm openai
```

To start the server, run:
```bash
vllm serve model --api-key EMPTY
```

Note that `model` in the command above refers to the directory which contains our model weights.

When running the server using `vllm serve`, the process does not run in the background so you will need to use another terminal session to invoke the model.

You can use the python script (`model_client.py`) to invoke the model, which allows you to pass in your own question and context:
```bash
python model_client.py --port 8000 --question "QUESTION" --context "CONTEXT"
```

If you invoke this script without providing the `--question` and `--context` arguments, an example from your test data will be used so you can familarize yourself with the output.

