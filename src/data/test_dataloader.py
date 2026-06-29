from src.data.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders(batch_size=8)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels)

print(f"\nTraining batches : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")