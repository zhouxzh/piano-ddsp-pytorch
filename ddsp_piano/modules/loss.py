import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np

class HybridLoss(nn.Module):
    def __init__(
        self,
        n_ffts,
        inharm,
        phase,
        weight=0.05,
        loss_type='L1',
        l1_weight_of_inharm=0.1,
        dry_weight=0.7,
        wet_weight=0.3,
        peak_weight=0.01,
        tail_weight=0.02,
        reverb_mode='ir',
    ):
        super().__init__()
        self.inharm = inharm
        self.phase = phase 

        self.mssLoss = MSSLoss(n_ffts)
        self.reverb_l1_loss = ReverbRegularizer(weight, loss_type)
        self.l1_weight_of_inharm = l1_weight_of_inharm
        self.dry_weight = float(dry_weight)
        self.wet_weight = float(wet_weight)
        self.peak_weight = float(peak_weight)
        self.tail_weight = float(tail_weight)
        self.reverb_mode = reverb_mode

    def components(self, y_pred, y_true, reverb_ir=None, dry_pred=None):
        """Return total, wet spectral, reverb, and dry spectral losses.

        The dry branch is deliberately part of the objective so the learned
        reverb cannot hide an under-trained oscillator behind a large IR.
        ``reverb_ir`` is an IR for the legacy model or a compact control vector
        for the v2 FDN path.
        """
        wet_loss = self.mssLoss(y_pred, y_true)
        if dry_pred is None:
            dry_pred = y_pred
        dry_loss = self.mssLoss(dry_pred, y_true)

        reverb_loss = y_pred.new_zeros(())
        if reverb_ir is not None and self.reverb_mode == 'ir':
            reverb_loss = self.reverb_l1_loss(reverb_ir)
            peak_penalty = torch.relu(reverb_ir.abs().amax(dim=-1) - 0.1).mean()
            tail_start = min(16_000, reverb_ir.shape[-1])
            tail_penalty = reverb_ir[..., tail_start:].abs().mean()
            reverb_loss = reverb_loss + self.peak_weight * peak_penalty
            reverb_loss = reverb_loss + self.tail_weight * tail_penalty
        elif reverb_ir is not None and self.reverb_mode == 'fdn':
            reverb_loss = 1e-3 * reverb_ir.square().mean()

        total = self.dry_weight * dry_loss + self.wet_weight * wet_loss + reverb_loss
        if not self.phase:
            l1_penalty = self.l1_weight_of_inharm * (
                self.inharm.slopes_modifier.abs().sum()
                + self.inharm.offsets_modifier.abs().sum()
            )
            total = total + l1_penalty
        return total, wet_loss, reverb_loss, dry_loss
    
    def forward(self, y_pred, y_true, reverb_ir):
        total, wet_loss, reverb_loss, _ = self.components(y_pred, y_true, reverb_ir)
        return total, wet_loss, reverb_loss


###### from ddsp-singing-vocoder
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7, name='SSSLoss'):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.name = name
    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
    
        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        window = torch.hann_window(self.n_fft, device=x_true.device, dtype=x_true.dtype)
        S_true = torch.stft(
            x_true,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ).abs()
        S_pred = torch.stft(
            x_pred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ).abs()
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return {'loss':loss}


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio = 1.0, overlap=0.75, eps=1e-7, use_reverb=True, name='MultiScaleLoss'):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        self.ratio = ratio
        self.name = name
    def forward(self, x_pred, x_true, return_spectrogram=True):
        x_pred = x_pred[..., :x_true.shape[-1]]
        if return_spectrogram:
            losses = []
            spec_true = []
            spec_pred = []
            for loss in self.losses:
                loss_dict = loss(x_true, x_pred)
                losses += [loss_dict['loss']]
        
        return self.ratio*sum(losses).sum()

##### reverb_loss
class ReverbRegularizer(nn.Module):
    """Regularization loss on the reverb impulse response.
    Params:
        - weight (float): loss weight.
        - loss_type {'L1', 'L2'}: compute L1 or L2 regularization.
    """
    def __init__(self, weight=0.01, loss_type='L1'):
        super(ReverbRegularizer, self).__init__()
        if loss_type not in {'L1', 'L2'}:
            raise ValueError(f"loss_type must be 'L1' or 'L2', got {loss_type!r}")
        self.weight = weight
        self.loss_type = loss_type
    def forward(self, reverb_ir):
        if self.loss_type == 'L1':
            loss = torch.sum(torch.abs(reverb_ir))
        elif self.loss_type == 'L2':
            loss = torch.sum(torch.square(reverb_ir))
        loss /= reverb_ir.shape[0] # Divide by batch size
        return self.weight * loss
