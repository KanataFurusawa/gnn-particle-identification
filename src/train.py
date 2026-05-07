import torch
import torch.nn.functional as F

# =========================
# train
# =========================
def train(model, loader, optimizer, device):

    model.train()

    total_loss = 0

    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)

        loss = F.cross_entropy(out, data.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# test
# =========================
def test(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()

            total += data.num_graphs

    return correct / total