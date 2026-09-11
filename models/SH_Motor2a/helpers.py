import os 
import torch 
import torch.nn.functional as F


#===decision ablations===
#always return in place output so extraction is handled externally
DECISIONS = {
    'sigmoid': torch.sigmoid, #is the function itself not the call
    'softmax': lambda x: F.softmax(x, dim=-1)
}

#===label ablations===
ASSUMPTION_1 = ["task-idle", "task-active"] #is just 'task-active' for sigmoid readout but both for softmax
ASSUMPTION_2 = ["left_hand", "right_hand", "both_feet", "tongue"] #paper and other study's label
ASSUMPTION_3 = ["hand", "foot", "tongue"] #isolating lateral expressibility

#===discrete threshold ablations===
DISCRETE_THRESHOLDS = [0.5, 0.8]

#===savers and loaders==
#generic over the decoder classes, every model carries its own config
def save_motor2a_model(model, optimiser, epoch, save_path, filename):
    os.makedirs(save_path, exist_ok=True) 

    torch.save({"config": model.config, "state_dict": model.state_dict(), "optimiser": optimiser.state_dict(),
                "epoch": epoch}, f"{save_path}/{filename}.pt")

def load_motor2a_model(path, model_cls, device=None):
    checkpoint = torch.load(path, map_location=device)
    model = model_cls(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model