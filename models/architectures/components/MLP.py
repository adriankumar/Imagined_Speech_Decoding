import torch.nn as nn
import torch 

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden=2, dropout=0.0, reduce_layers=True):
        super().__init__()
        assert n_hidden >= 1, "expected at least one hidden layer"

        self._input_dim = input_dim
        self._output_dim = output_dim

        if reduce_layers:
            layer_dims = self._hidden_dims(hidden_dim, output_dim, n_hidden)
        else:
            layer_dims = [hidden_dim] * n_hidden #all hidden layers the same size up to output

        self._layers = nn.Sequential()
        current_dim = input_dim

        for width in layer_dims:
            self._layers.append(nn.Linear(current_dim, width))
            self._layers.append(nn.GELU()) #using GELU

            if dropout > 0.0:
                self._layers.append(nn.Dropout(dropout))

            current_dim = width

        self._layers.append(nn.Linear(current_dim, output_dim))

    #computes reduced hidden dims per layer
    def _hidden_dims(self, hidden_dim, output_dim, n_hidden):
        if n_hidden == 1:
            return [hidden_dim]

        ratio = (output_dim / hidden_dim) ** (1.0 / n_hidden) #(out / hidden)**(1/n) 
        return [max(output_dim, int(round(hidden_dim * ratio ** i))) for i in range(n_hidden)]

    #input expected to be a vector, B x input_dim
    def forward(self, x):
        assert x.shape[-1] == self._input_dim, "input size expected to match input_dim"
        return self._layers(x) #B x output_dim; raw logits

    def predict(self, x, threshold=0.5):
        probs = torch.sigmoid(self.forward(x)) #B x output_dim; independent probabilities from 0-1
        return probs, probs > threshold #return both the probabilities, and the classes as True/False based on a threshold
    
    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")