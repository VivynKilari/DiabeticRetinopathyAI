import torch

from src.data.dataloader import create_dataloaders
from src.models.model import DRClassifier
from src.training.engine import train_one_epoch, validate
from src.training.losses import get_loss_function
from src.training.metrics import calculate_metrics
from src.utils.config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
)

model = DRClassifier().to(DEVICE)

train_loader, val_loader = create_dataloaders()

criterion = get_loss_function()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_accuracy = 0

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE,
    )

    val_loss, preds, labels = validate(
        model,
        val_loader,
        criterion,
        DEVICE,
    )

    metrics = calculate_metrics(
        labels,
        preds,
    )

    print()

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Validation Loss : {val_loss:.4f}")

    print(metrics)

    if metrics["accuracy"] > best_accuracy:

        best_accuracy = metrics["accuracy"]

        torch.save(
            model.state_dict(),
            "outputs/models/best_model.pth",
        )

        print("✅ Best model saved.")