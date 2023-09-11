import argparse

import gradio as gr
import torch
from transformers import XLMTokenizer
from model.transformer.DecoderOnly import DecoderOnlyTransformer
from model.lstm.lstm import LSTMTextGenerator
from datetime import datetime

tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
vocab_size = tokenizer.vocab_size

transdecoder = None
lstm_text_generator = None


def greet(model, name, seed, temp, length, strategy, top_k, top_p):
    if seed == -1:
        seed = datetime.now().timestamp()
    torch.manual_seed(seed)

    tokenized = tokenizer.encode(name)[:-1]  # remove </s> token

    top_k = int(top_k)

    if model == "Transformer-Decoder":
        output = transdecoder.generate(
            tokenized,
            length=int(length),
            temperature=temp,
            strategy=strategy,
            top_k=top_k,
            top_p=top_p,
        )
    elif model == "LSTM":
        output = lstm_text_generator.generate(
            tokenized, length=int(length), temperature=temp
        )

    return tokenizer.decode(output)


demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Dropdown(
            ["Transformer-Decoder", "LSTM"], value="Transformer-Decoder", label="Model"
        ),
        gr.Textbox(value="Dawno, dawno temu, w odległej krainie", label="Input text"),
        gr.Number(
            value=-1,
            minimum=-1,
            maximum=100_000,
            label="Seed",
            info="Set to -1 for random seed",
        ),
        gr.Slider(minimum=0.05, maximum=10, step=0.01, value=1, label="Temperature"),
        gr.Number(value=100, minimum=10, maximum=100_000, label="Length"),
        gr.Dropdown(
            label="Strategy",
            choices=["top_k_top_p", "greedy", "multinomial"],
            value="top_k_top_p",
        ),
        gr.Number(
            minimum=0, maximum=tokenizer.vocab_size, step=1, value=0, label="Top-k"
        ),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Output text", scale=2),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    parser.add_argument("--transformer-embedding", type=int, default=100)
    parser.add_argument("--transformer-block_size", type=int, default=128)
    parser.add_argument("--transformer-head", type=int, default=8)
    parser.add_argument("--transformer-layer", type=int, default=8)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)

    parser.add_argument("--vocab-size", type=int, default=tokenizer.vocab_size)
    parser.add_argument("--lstm-embedding_size", type=int, default=100)
    parser.add_argument("--lstm-layers", type=int, default=3)
    parser.add_argument("--lstm-bidirectional", type=bool, default=True)
    parser.add_argument("--lstm-dropout", type=float, default=0.5)
    parser.add_argument("--lstm-size", type=int, default=256)
    parser.add_argument("--lstm-sequence_length", type=int, default=128)

    parser.add_argument("--gpu", action="store_false", help="Use GPU if available")

    parser.add_argument("--transformer-model-path", type=str, required=True)
    parser.add_argument("--lstm-model-path", type=str, required=True)

    args = parser.parse_args()

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    transdecoder = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        n_embd=args.transformer_embedding,
        n_head=args.transformer_head,
        n_layer=args.transformer_layer,
        block_size=args.transformer_block_size,
        dropout=args.transformer_dropout,
    )
    transdecoder.load_state_dict(
        torch.load(args.transformer_model_path, map_location=device)
    )
    transdecoder.eval()
    transdecoder.to(device)

    lstm_text_generator = LSTMTextGenerator(
        vocab_size=vocab_size,
        emb_size=args.lstm_embedding_size,
        lstm_size=args.lstm_size,
        lstm_layers=args.lstm_layers,
        lstm_bidirectional=args.lstm_bidirectional,
        lstm_dropout=args.lstm_dropout,
        seq_len=args.lstm_sequence_length,
    )
    lstm_text_generator.load_state_dict(
        torch.load(args.lstm_model_path, map_location=device)
    )
    lstm_text_generator.eval()
    lstm_text_generator.to(device)

    demo.launch(server_name=args.host, share=args.share, server_port=args.port)
