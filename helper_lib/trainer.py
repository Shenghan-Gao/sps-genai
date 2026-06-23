import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints"
):
    best_val_accuracy = 0.0

    model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=120
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            avg_train_loss = running_loss / (progress_bar.n + 1)
            train_accuracy = 100 * running_correct / running_total

            progress_bar.set_postfix({
                "loss": f"{avg_train_loss:.4f}",
                "acc": f"{train_accuracy:.2f}%"
            })

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / running_total

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best"
            )

            print(f"New best model saved. Validation Accuracy: {val_accuracy:.2f}%")

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Accuracy={train_accuracy:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Accuracy={val_accuracy:.2f}%"
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    return model