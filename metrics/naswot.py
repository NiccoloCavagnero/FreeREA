import torch
import torch.nn as nn


def compute_naswot_score(net: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device):
    with torch.no_grad():
        codes = []

        def hook(self: nn.Module, m_input: torch.Tensor, m_output: torch.Tensor):
            code = (m_output > 0).flatten(start_dim=1)
            codes.append(code)

        hooks = []
        for m in net.modules():
            if isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(hook))

        _ = net(inputs)

        for h in hooks:
            h.remove()

        full_code = torch.cat(codes, dim=1)

        # Fast Hamming distance matrix computation
        del codes, _
        full_code_float = full_code.float()
        k = full_code_float @ full_code_float.t()
        del full_code_float
        not_full_code_float = torch.logical_not(full_code).float()
        k += not_full_code_float @ not_full_code_float.t()
        del not_full_code_float

        return torch.slogdet(k).logabsdet.item()
