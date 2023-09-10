# bajkogenerator

Generating fairy tales in Polish using LSTM and Transformer neural networks.

Project for PIAT 2023-Summer course at Politechnika Łódzka. 

## Docker

### Build
To build docker image run from project root directory
```
docker build -t bajkogenerator:latest -f ./docker/Dockerfile .
```

### Run
To launch docker image with jupyter session run 
```
docker run -p 8888:8888 bajkogenerator:latest jupyter
```

To launch docker image for inference with gradio run 
```
docker run -p 7860:7860 bajkogenerator:latest gradio
```

To launch docker image for training lstm model run 
```
docker run bajkogenerator:latest train_lstm
```

To launch docker image for training transformer model run 
```
docker run bajkogenerator:latest train_transformer
```

To train models with dataset mounted add `-v {path_to_dataset}:/dataset` to docker run command.

Example how to run docker image for training transformer model with dataset mounted:
```
docker run -v {path_to_dataset}:/app/dataset bajkogenerator:latest train_transformer --train-data-path /dataset
```

To use GPU in docker container add `--gpus all` to docker run command.

Example how to run inference docker image with GPU:
```
docker run --gpus all -p 7860:7860 -v {path_to_weights}:/app/weights bajkogenerator:latest gradio --transformer-model-path=/app/weights/{transformer_weights} --lstm-model-path /app/weights/{lstm_weights}
```

## Dataset

Dataset used in training is available at [Google drive folder](https://drive.google.com/drive/folders/1BxkEWUUNQN78A1AG-pTnqGR7C6XXJhAr).

## Weights

Weights of trained models are available at [Google drive folder](https://drive.google.com/drive/folders/19mm4wb00qa2q-Km5ceEuahpGMi-QNb9f).