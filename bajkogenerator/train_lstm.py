import argparse

import torch
from tqdm import tqdm

from model.lstm.lstm import LSTMTextGenerator
from data.story_dataloader import StoryDataset


def train(
    embedding_size,
    lstm_size,
    layers,
    bidirectional,
    dropout,
    seq_len,
    epochs,
    batch_size,
    save_path,
    train_data_path,
    vocab_size,
    device,
    lr,
    num_workers,
    checkpoint_interval,
    silent,
):
    lstm = LSTMTextGenerator(
        vocab_size=vocab_size,
        emb_size=embedding_size,
        lstm_size=lstm_size,
        lstm_layers=layers,
        lstm_bidirectional=bidirectional,
        lstm_dropout=dropout,
        seq_len=seq_len,
    )
    lstm.to(device)

    optimizer = torch.optim.RAdam(lstm.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    data_loader = StoryDataset(
        train_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        sequence_size=seq_len,
    )

    if not silent:
        print("Starting training with following parameters:")
        print(f"embedding_size: {embedding_size}")
        print(f"lstm_size: {lstm_size}")
        print(f"layers: {layers}")
        print(f"bidirectional: {bidirectional}")
        print(f"dropout: {dropout}")
        print(f"seq_len: {seq_len}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"save_path: {save_path}")
        print(f"train_data_path: {train_data_path}")
        print(f"vocab_size: {vocab_size}")
        print(f"device: {device}")
        print(f"lr: {lr}")
        print(f"num_workers: {num_workers}")
        print(f"checkpoint_interval: {checkpoint_interval}")
        print("--------------------------")

    for epoch in tqdm(range(1, epochs + 1)):
        running_loss = 0.0

        for batch in tqdm(data_loader):
            batch_tensor = torch.LongTensor(batch).to(device)

            x_batch = batch_tensor[:, :seq_len]
            y_batch = batch_tensor[:, seq_len]

            lstm.zero_grad()

            output = lstm(x_batch)

            loss = criterion(output, y_batch)

            running_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 0.5)

            optimizer.step()

        if not silent:
            print(f"Epoch {epoch}, loss: {running_loss}")

        if epoch % checkpoint_interval == 0:
            save_name = (
                f"embedding_size_{embedding_size}_"
                f"size_{lstm_size}_"
                f"layers_{layers}_"
                f"bidirectional_{bidirectional}_"
                f"dropout_{dropout}_"
                f"seq_len_{seq_len}_"
                f"epochs_{epoch}_"
                f"class_{lstm.__class__.__name__}.pth"
            )
            torch.save(lstm.state_dict(), save_path + save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embedding_size", type=int, default=100, help="Embedding size"
    )
    parser.add_argument("--lstm-size", type=int, default=254, help="LSTM size")
    parser.add_argument("--n-layer", type=int, default=2, help="Number of layers")
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-path", type=str, default="./")
    parser.add_argument(
        "--train-data-path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument("--vocab-size", type=int, default=50560, help="Vocab size")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Interval between checkpoints",
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Suppresses text output"
    )

    args = parser.parse_args()

    device_arg = (
        torch.device(args.gpus if torch.cuda.is_available() else "cpu")
        if args.gpu
        else torch.device("cpu")
    )

    train(
        embedding_size=args.embedding_size,
        lstm_size=args.lstm_size,
        layers=args.n_layer,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        train_data_path=args.train_data_path,
        vocab_size=args.vocab_size,
        device=device_arg,
        lr=args.lr,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        silent=args.silent,
    )
