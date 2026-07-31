import torch
import torch.nn as nn 
from torch.nn import functional as F

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
        reverb_regularizer_reduction="sum_per_sample",
        energy_weight=0.0,
        onset_weight=0.0,
        centroid_weight=0.0,
        tail_weight_audio=0.0,
        energy_hard_fraction=0.0,
        sample_rate=16_000,
        frame_rate=250,
        loss_version="legacy",
        component_scales=None,
        spectral_layout="separate",
    ):
        super().__init__()
        self.inharm = inharm
        self.phase = phase 

        self.mssLoss = MSSLoss(n_ffts, transform_layout=spectral_layout)
        self.reverb_l1_loss = ReverbRegularizer(
            weight,
            loss_type,
            reduction=reverb_regularizer_reduction,
        )
        self.l1_weight_of_inharm = l1_weight_of_inharm
        self.dry_weight = float(dry_weight)
        self.wet_weight = float(wet_weight)
        self.peak_weight = float(peak_weight)
        self.tail_weight = float(tail_weight)
        self.reverb_mode = reverb_mode
        self.energy_weight = float(energy_weight)
        self.onset_weight = float(onset_weight)
        self.centroid_weight = float(centroid_weight)
        self.tail_weight_audio = float(tail_weight_audio)
        self.energy_hard_fraction = float(energy_hard_fraction)
        if loss_version not in {"legacy", "perceptual_v2"}:
            raise ValueError(f"unsupported loss_version: {loss_version}")
        self.loss_version = loss_version
        self.component_scales = {
            "wet": 1.0,
            "energy": 1.0,
            "onset": 1.0,
            "centroid": 1.0,
            "tail": 1.0,
        }
        if component_scales is not None:
            self.component_scales.update(
                {name: float(value) for name, value in component_scales.items()}
            )
        self.energy_window = max(1, int(round(sample_rate * 0.064)))
        self.onset_window = max(1, int(round(sample_rate * 0.016)))
        self.envelope_hop = max(1, int(round(sample_rate * 0.004)))
        self.samples_per_frame = sample_rate // frame_rate
        self.tail_frames = max(1, int(round(frame_rate * 0.5)))
        self.register_buffer(
            "centroid_window",
            torch.hann_window(1024),
            persistent=False,
        )

    @staticmethod
    def _frame_rms(value, window, hop):
        value = value.unsqueeze(1) if value.ndim == 2 else value
        power = F.avg_pool1d(value.square(), window, stride=hop, ceil_mode=True)
        return torch.sqrt(power.clamp_min(1e-8)).squeeze(1)

    def _energy_loss(self, prediction, target):
        pred_rms = self._frame_rms(prediction, self.energy_window, self.energy_window)
        target_rms = self._frame_rms(target, self.energy_window, self.energy_window)
        errors = torch.abs(
            torch.log1p(100.0 * pred_rms) - torch.log1p(100.0 * target_rms)
        )
        return self._robust_mean(errors, self.energy_hard_fraction)

    @staticmethod
    def _robust_mean(errors, hard_fraction):
        mean = errors.mean()
        fraction = float(hard_fraction)
        if fraction <= 0.0:
            return mean
        count = max(1, int(round(errors.numel() * min(fraction, 1.0))))
        hard = torch.topk(errors.reshape(-1), count, sorted=False).values.mean()
        return 0.5 * mean + 0.5 * hard

    def _onset_loss(self, prediction, target):
        pred = self._frame_rms(prediction, self.onset_window, self.envelope_hop)
        truth = self._frame_rms(target, self.onset_window, self.envelope_hop)
        pred = pred / pred.mean(dim=-1, keepdim=True).clamp_min(1e-5)
        truth = truth / truth.mean(dim=-1, keepdim=True).clamp_min(1e-5)
        return F.l1_loss(torch.diff(pred, dim=-1), torch.diff(truth, dim=-1))

    def _centroid_loss(self, prediction, target):
        minimum = min(prediction.shape[-1], target.shape[-1])
        combined = torch.cat((prediction[..., :minimum], target[..., :minimum]), dim=0)
        magnitude = torch.stft(
            combined,
            n_fft=1024,
            hop_length=256,
            window=self.centroid_window,
            return_complex=True,
        ).abs()
        prediction_magnitude, target_magnitude = magnitude.split(prediction.shape[0], dim=0)
        frequencies = torch.linspace(
            0.0,
            1.0,
            prediction_magnitude.shape[-2],
            device=prediction.device,
            dtype=prediction_magnitude.dtype,
        ).view(1, -1, 1)

        def centroid(value):
            return (value * frequencies).sum(dim=-2) / value.sum(dim=-2).clamp_min(1e-7)

        errors = torch.abs(
            centroid(prediction_magnitude) - centroid(target_magnitude)
        )
        active = (target_magnitude.sum(dim=-2) > 1e-5).to(errors.dtype)
        return (errors * active).sum() / active.sum().clamp_min(1.0)

    def _tail_decay_loss(self, prediction, target, conditioning):
        if conditioning is None:
            return prediction.new_zeros(())
        active = conditioning[..., 0].ne(0).any(dim=-1)
        frame_count = min(active.shape[-1], prediction.shape[-1] // self.samples_per_frame)
        if frame_count < 2:
            return prediction.new_zeros(())
        active = active[..., :frame_count]
        indices = torch.arange(frame_count, device=active.device).view(1, -1)
        last_active = torch.cummax(
            torch.where(active, indices, torch.full_like(indices, -frame_count)), dim=-1
        ).values
        distance = indices - last_active
        mask = (~active) & (last_active >= 0) & (distance > 0) & (distance <= self.tail_frames)
        pred_rms = self._frame_rms(
            prediction[..., : frame_count * self.samples_per_frame],
            self.samples_per_frame,
            self.samples_per_frame,
        )[..., :frame_count]
        target_rms = self._frame_rms(
            target[..., : frame_count * self.samples_per_frame],
            self.samples_per_frame,
            self.samples_per_frame,
        )[..., :frame_count]
        errors = F.smooth_l1_loss(
            torch.log1p(1000.0 * pred_rms),
            torch.log1p(1000.0 * target_rms),
            reduction="none",
        )
        weights = mask.to(errors.dtype)
        return (errors * weights).sum() / weights.sum().clamp_min(1.0)

    def components(
        self,
        y_pred,
        y_true,
        reverb_ir=None,
        dry_pred=None,
        conditioning=None,
    ):
        """Return total plus wet/reverb/dry/energy/onset/centroid/tail losses.

        The dry branch is deliberately part of the objective so the learned
        reverb cannot hide an under-trained oscillator behind a large IR.
        ``reverb_ir`` is an IR for the legacy model or a compact control vector
        for the v2 FDN path.
        """
        wet_loss = self.mssLoss(y_pred, y_true)
        if dry_pred is None:
            dry_pred = y_pred
        dry_loss = (
            self.mssLoss(dry_pred, y_true)
            if self.loss_version == "legacy"
            else y_pred.new_zeros(())
        )

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

        energy_loss = y_pred.new_zeros(())
        onset_loss = y_pred.new_zeros(())
        comparison_signal = dry_pred if self.loss_version == "legacy" else y_pred
        if self.energy_weight:
            energy_loss = self._energy_loss(comparison_signal, y_true)
        if self.onset_weight:
            onset_loss = self._onset_loss(comparison_signal, y_true)
        centroid_loss = y_pred.new_zeros(())
        tail_loss = y_pred.new_zeros(())
        if self.centroid_weight:
            centroid_loss = self._centroid_loss(y_pred, y_true)
        if self.tail_weight_audio:
            tail_loss = self._tail_decay_loss(y_pred, y_true, conditioning)
        total = reverb_loss
        if self.loss_version == "legacy":
            total = total + self.dry_weight * dry_loss + self.wet_weight * wet_loss
        else:
            total = total + self.wet_weight * self.component_scales["wet"] * wet_loss
        total = total + (
            self.energy_weight * self.component_scales["energy"] * energy_loss
            + self.onset_weight * self.component_scales["onset"] * onset_loss
            + self.centroid_weight * self.component_scales["centroid"] * centroid_loss
            + self.tail_weight_audio * self.component_scales["tail"] * tail_loss
        )
        if not self.phase:
            l1_penalty = self.l1_weight_of_inharm * (
                self.inharm.slopes_modifier.abs().sum()
                + self.inharm.offsets_modifier.abs().sum()
            )
            total = total + l1_penalty
        return (
            total,
            wet_loss,
            reverb_loss,
            dry_loss,
            energy_loss,
            onset_loss,
            centroid_loss,
            tail_loss,
        )

    @staticmethod
    def velocity_monotonic_loss(
        low_amplitudes,
        high_amplitudes,
        active_mask,
        margin=0.01,
    ):
        """Penalize raw amplitude controls that fall as MIDI velocity rises."""
        low = F.softplus(low_amplitudes)
        high = F.softplus(high_amplitudes)
        mask = active_mask.to(dtype=low.dtype)
        penalty = F.relu(low + float(margin) - high) * mask
        return penalty.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def velocity_response_loss(
        low_amplitudes,
        high_amplitudes,
        active_mask,
        expected_log_ratio,
    ):
        """Match the predicted amplitude ratio to a calibrated MIDI response."""
        low = F.softplus(low_amplitudes).clamp_min(1e-7)
        high = F.softplus(high_amplitudes).clamp_min(1e-7)
        predicted = torch.log(high) - torch.log(low)
        mask = active_mask.to(dtype=predicted.dtype)
        errors = F.smooth_l1_loss(predicted, expected_log_ratio, reduction="none") * mask
        return errors.sum() / mask.sum().clamp_min(1.0)
    
    def forward(self, y_pred, y_true, reverb_ir):
        total, wet_loss, reverb_loss, _, _, _, _, _ = self.components(
            y_pred, y_true, reverb_ir
        )
        return total, wet_loss, reverb_loss


###### from ddsp-singing-vocoder
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(
        self,
        n_fft=111,
        alpha=1.0,
        overlap=0.75,
        eps=1e-7,
        name='SSSLoss',
        transform_layout="separate",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.name = name
        if transform_layout not in {"separate", "combined"}:
            raise ValueError("transform_layout must be 'separate' or 'combined'")
        self.transform_layout = transform_layout
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
            persistent=False,
        )
    def forward(self, x_true, x_pred):
        min_len = min(x_true.shape[-1], x_pred.shape[-1])
    
        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        if self.transform_layout == "combined":
            batch_size = x_true.shape[0]
            combined = torch.cat((x_true, x_pred), dim=0)
            spectra = torch.stft(
                combined,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
            ).abs()
            S_true, S_pred = spectra.split(batch_size, dim=0)
        else:
            S_true = torch.stft(
                x_true,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
            ).abs()
            S_pred = torch.stft(
                x_pred,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
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

    def __init__(
        self,
        n_ffts,
        alpha=1.0,
        ratio=1.0,
        overlap=0.75,
        eps=1e-7,
        use_reverb=True,
        name='MultiScaleLoss',
        transform_layout="separate",
    ):
        super().__init__()
        self.losses = nn.ModuleList(
            [
                SSSLoss(
                    n_fft,
                    alpha,
                    overlap,
                    eps,
                    transform_layout=transform_layout,
                )
                for n_fft in n_ffts
            ]
        )
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
    def __init__(
        self,
        weight=0.01,
        loss_type='L1',
        reduction="sum_per_sample",
    ):
        super(ReverbRegularizer, self).__init__()
        if loss_type not in {'L1', 'L2'}:
            raise ValueError(f"loss_type must be 'L1' or 'L2', got {loss_type!r}")
        if reduction not in {"sum_per_sample", "mean"}:
            raise ValueError(
                "reduction must be 'sum_per_sample' or 'mean', "
                f"got {reduction!r}"
            )
        self.weight = weight
        self.loss_type = loss_type
        self.reduction = reduction
    def forward(self, reverb_ir):
        if self.loss_type == 'L1':
            values = torch.abs(reverb_ir)
        elif self.loss_type == 'L2':
            values = torch.square(reverb_ir)
        if self.reduction == "mean":
            loss = values.mean()
        else:
            loss = values.sum() / reverb_ir.shape[0]
        return self.weight * loss
