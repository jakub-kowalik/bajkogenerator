#!/bin/bash

# Check the first argument (if provided)
if [ -n "$1" ]; then
    case "$1" in
        "jupyter")
            exec jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
            ;;
        "gradio")
            exec python3 /app/bajkogenerator/gradio_text_generation.py "${@:2}"
            ;;
        "train_lstm")
            exec python3 /app/bajkogenerator/train_lstm.py "${@:2}"
            ;;
        "train_transformer")
            exec python3 /app/bajkogenerator/train_decoder_only_transformer.py "${@:2}"
            ;;
        *)
            echo "Unknown command: $1"
            ;;
    esac
else
    echo "No command provided"
fi
