import torch
from ds import DM
from mr import MyModel
from torch.utils.data import DataLoader, random_split
import random
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
batch_size = 1024
accumulation_steps = 1

full_dataset = DM()
total_len = len(full_dataset)

train_len = int(0.75 * total_len)
val_len = int(0.13 * total_len)
test_len = total_len - train_len - val_len
classes = ["Class0", "Class1", "Class2", "Class3"]

random_int = random.randint(0, 9999)  # 7979
print(f"Using random seed: {random_int}")

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(random_int),
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
val_losses = []
val_accuracies = []

model = MyModel(input_channel=2, num_classes=len(classes)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


# 验证函数
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    model.train()
    return avg_loss, acc


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 测试集评估 + 混淆矩阵可视化
def test_and_plot_confusion(
    model, dataloader, class_names=None, save_path="confusion_matrix.png"
):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    acc = (all_preds_tensor == all_labels_tensor).float().mean().item()

    all_preds_list = all_preds_tensor.tolist()
    all_labels_list = all_labels_tensor.tolist()

    cm = confusion_matrix(all_labels_list, all_preds_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Test Confusion matrix saved as {save_path}. Accuracy: {acc * 100:.2f}%")
    model.train()


# Early stopping 参数
best_val_acc = 0.0
best_val_loss = float("inf")
patience = 100
no_improve_count = 0

# 训练循环
for epoch in range(1000):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast(device_type="cuda", dtype=dtype):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(
                f"Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}",
                end="\r",
            )

    # 每个 epoch 后验证
    val_loss, val_acc = evaluate(model, val_loader)
    print(
        f"✅ Epoch [{epoch+1}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.3f}%"
    )

    # 保存最佳模型（按准确率）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model, "model_checkpoints/best_model_acc.pth")
        print(
            f"📌 New best accuracy: {best_val_acc*100:.3f}%, Val Loss: {best_val_loss:.4f} — model saved."
        )
        test_and_plot_confusion(
            model,
            test_loader,
            class_names=classes,
            save_path=f"confusion_matrix_epoch_{epoch+1}.png",
        )
        no_improve_count = 0
    else:
        no_improve_count += 1
        if epoch % 50 == 0:
            test_and_plot_confusion(
                model,
                test_loader,
                class_names=classes,
                save_path=f"confusion_matrix_epoch_{epoch+1}.png",
            )

    torch.save(model, f"model_checkpoints/last_model.pth")
    train_losses.append(loss.item())  # 记录最后一个 batch 的 loss（可改为平均）
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Early stopping 检查
    if no_improve_count >= patience and best_val_acc >= 0.95:
        print(
            f"⏹️ Early stopping triggered at epoch {epoch+1}. No improvement for {no_improve_count} epochs."
        )
        break

    # 每隔 50 个 epoch 保存一次快照
    if (epoch + 1) % 50 == 0:
        torch.save(model, f"model_checkpoints/model_epoch_{epoch+1}.pth")


# 加载最佳模型并评估测试集
model = (
    torch.load("model_checkpoints/best_model_acc.pth", weights_only=False)
    .to(device)
    .eval()
)
test_and_plot_confusion(model, test_loader, class_names=classes)

model = (
    torch.load("model_checkpoints/last_model.pth", weights_only=False).to(device).eval()
)
test_and_plot_confusion(
    model,
    test_loader,
    class_names=classes,
    save_path="confusion_matrix_last.png",
)

epochs = list(range(1, len(val_accuracies) + 1))

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：Loss 曲线
ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue", linestyle="--")
ax1.plot(epochs, val_losses, label="Val Loss", color="tab:orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.legend(loc="upper left")
ax1.grid(True)

# 右轴：Accuracy 曲线
ax2 = ax1.twinx()
ax2.plot(
    epochs,
    [acc * 100 for acc in val_accuracies],
    label="Val Accuracy",
    color="tab:green",
)
ax2.set_ylabel("Accuracy (%)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.legend(loc="upper center")

plt.title("Training Loss & Validation Accuracy Over Epochs")
plt.tight_layout()
plt.savefig("loss_accuracy_combined.png")
print("📊 Combined plot saved as loss_accuracy_combined.png")
