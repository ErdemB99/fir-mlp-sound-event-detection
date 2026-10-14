
import torch
import torch.nn as nn
import torch.nn.functional as F


class FIR_MLP(nn.Module):
    """
    Temporal FIR-MLP for multi-label sound event detection.

    Architecture:
        Input: 128 features
        Hidden: 64 neurons
        Output: 11 sound event classes

    FIR filter lengths:
        Input -> Hidden: 5
        Hidden -> Output: 3
    """

    def __init__(
        self,
        num_layers,
        num_neurons,
        FIR_lengths,
        learning_rate=0.01,
        momentum=0.9
    ):
        super().__init__()

        if num_layers != 3 or len(num_neurons) != 3:
            raise ValueError(
                "Expected exactly 3 layers: input, hidden, output."
            )

        if len(FIR_lengths) == 2:
            fir_input, fir_hidden = FIR_lengths
        elif len(FIR_lengths) >= 3:
            fir_input, fir_hidden = FIR_lengths[1:3]
        else:
            raise ValueError("Two FIR lengths are required.")

        if fir_input < 1 or fir_hidden < 1:
            raise ValueError("FIR lengths must be positive.")

        self.num_layers = num_layers
        self.num_neurons = list(num_neurons)
        self.fir_input = int(fir_input)
        self.fir_hidden = int(fir_hidden)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)

        input_dim, hidden_dim, output_dim = num_neurons

        self.hidden_layer = nn.Linear(
            input_dim * self.fir_input,
            hidden_dim
        )

        self.output_layer = nn.Linear(
            hidden_dim * self.fir_hidden,
            output_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def create_temporal_windows(x, fir_length):
        """
        Input:  [batch, time, features]
        Output: [batch, time, features * fir_length]

        At time t, use x[t], x[t-1], ..., x[t-K+1].
        Missing past frames are zero-padded.
        """
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, time, features]."
            )

        batch_size, time_steps, features = x.shape

        x_padded = F.pad(
            x,
            pad=(0, 0, fir_length - 1, 0),
            mode="constant",
            value=0.0
        )

        windows = x_padded.unfold(
            dimension=1,
            size=fir_length,
            step=1
        )

        # [batch, time, features, FIR]
        # Newest temporal tap first.
        windows = windows.flip(dims=[-1])

        return windows.contiguous().reshape(
            batch_size,
            time_steps,
            features * fir_length
        )

    def forward(self, x):
        """
        Input:  [batch, frames, input_features]
        Output: [batch, frames, output_classes]

        Returns logits for BCEWithLogitsLoss.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input, got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.num_neurons[0]:
            raise ValueError(
                f"Expected {self.num_neurons[0]} features, "
                f"got {x.shape[-1]}"
            )

        input_windows = self.create_temporal_windows(
            x, self.fir_input
        )

        hidden = torch.sigmoid(
            self.hidden_layer(input_windows)
        )

        hidden_windows = self.create_temporal_windows(
            hidden, self.fir_hidden
        )

        logits = self.output_layer(hidden_windows)

        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).int()

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    model = FIR_MLP(
        num_layers=3,
        num_neurons=[128, 64, 11],
        FIR_lengths=[5, 3]
    )

    x = torch.randn(2, 100, 128)
    y = torch.randint(0, 2, (2, 100, 11)).float()

    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    assert logits.shape == (2, 100, 11)

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    print("Parameters:", model.count_parameters())
    print("Output shape:", tuple(logits.shape))
    print("BCE loss:", loss.item())
    print("Forward/backward smoke test passed.")
