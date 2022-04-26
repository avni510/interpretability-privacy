from captum.attr import IntegratedGradients, NoiseTunnel
import torch

def smooth_grad(model, input_val, target_class):
    input_val = input_val.clone()
    input_val = torch.unsqueeze(input_val, 0)
    input_val.requires_grad = True

    model.eval()

    int_grad = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(int_grad)
    attribution = noise_tunnel.attribute(
            input_val,
            nt_type='smoothgrad',
            nt_samples=5,
            target=target_class
            )
    return (input_val.detach(), attribution.detach(), target_class)
