from architecture_components import CTM 
import torch 

batch=4
input_dim = 40 #should be same as the embedding dim used in the attention and the action sync vector, unless you explicitly set vdim kdim 

model = CTM() #using default configs

input_features = torch.rand(batch, 3, input_dim)
hidden_states = None #will be a dictionary

predictions, hidden_states = model(input_features=input_features, neural_states=hidden_states)

print(f"prediction output size: {predictions.shape}")
pc = model.get_parameter_counts()

print(f"ctm parameter count:",
      f"trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")

