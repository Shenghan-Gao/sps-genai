import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader, get_train_val_loaders
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64
    epochs = 3
    learning_rate = 0.001

    train_loader, val_loader = get_train_val_loaders(
        data_dir="data",
        batch_size=batch_size,
        val_split=0.1
    )

    test_loader = get_data_loader(
        data_dir="data",
        batch_size=batch_size,
        train=False
    )

    model = get_model("CNN").to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        checkpoint_dir="checkpoints"
    )

    test_loss, test_accuracy = evaluate_model(
        trained_model,
        test_loader,
        criterion,
        device=device
    )

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()