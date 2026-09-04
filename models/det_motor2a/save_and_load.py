import os 
import torch 

#===savers and loaders==
#generic over the decoder classes, every model carries its own config
def save_motor2a_model(model, optimiser, epoch, save_path, filename):
    os.makedirs(save_path, exist_ok=True)

    torch.save({"config": model.config, "state_dict": model.state_dict(), "optimiser": optimiser.state_dict(),
                "epoch": epoch}, f"{save_path}/{filename}_{epoch}.pt")


def load_motor2a_model(path, model_cls, device=None):
    checkpoint = torch.load(path, map_location=device)
    model = model_cls(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model