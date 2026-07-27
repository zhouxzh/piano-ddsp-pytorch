import torch
import torch.nn as nn 

class PianoModel(nn.Module):
    """DDSP model for piano synthesis from MIDI conditioning.
    Args:
        - z_encoder: one-hot piano model embeddings.
        - note_release: extend active pitch conditioning.
        - context_network: context vector computation model from
        global inputs.
        - parallelizer: layer managing polyphony and batch axis
        merge and unmerge.
        - monophonic_network: monophonic string model as
        neural network.
        - inharm_model: inharmonicity model over tessitura.
        - detuner: tuning model for pitch to absolute f0
        frequency.
        - reverb_model: recording environment impulse responses.
    """
    def __init__(
        self,
        n_synths=16,
        z_encoder=None,
        note_release=None,
        context_network=None,
        parallelizer=None,
        monophonic_network=None,
        inharm_model=None,
        detuner=None,
        reverb_model=None,
        harmonic_synthesizer=None,
        noise_synthesizer=None,
        reverb_module=None,
        synthesis_layout="serial"):
        super(PianoModel, self).__init__()
        self.n_synths = n_synths
        self.z_encoder = z_encoder # num params ok 
        self.note_release = note_release # num params ok 
        self.context_network = context_network # num params ok
        self.parallelizer = parallelizer # num params ok
        self.monophonic_network = monophonic_network # num params ok
        self.inharm_model = inharm_model # num params ok
        self.detuner = detuner # num params ok
        self.harmonic_synthesizer = harmonic_synthesizer # num params ok
        self.noise_synthesizer = noise_synthesizer # num params ok
        self.reverb_module = reverb_module  # num params ok
        self.reverb_model = reverb_model # num params ok
        self.set_synthesis_layout(synthesis_layout)

    def set_synthesis_layout(self, layout):
        """Select the training-only polyphonic DSP execution layout."""
        if layout not in {"serial", "vectorized"}:
            raise ValueError("synthesis_layout must be 'serial' or 'vectorized'")
        self.synthesis_layout = layout

    def set_detune_enabled(self, enabled=True):
        """Control detuning independently from the training parameter scope."""
        self.detuner.use_detune = bool(enabled)

    def configure_training_stage(self, stage="controls"):
        """Configure trainable parameters for one explicit training stage.

        Detuning is deliberately independent from trainability.  In particular,
        ``refine`` and ``calibrate`` use the learned pitch model while updating
        only the main control path.
        """
        if stage not in {"controls", "pitch", "refine", "calibrate"}:
            raise ValueError(f"unsupported training stage: {stage}")

        for parameter in self.parameters():
            parameter.requires_grad = False

        pitch_stage = stage == "pitch"
        if pitch_stage:
            pitch_modules = [self.inharm_model, self.detuner]
            for module in pitch_modules:
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = True
            self.z_encoder.inharm_embedding.weight.requires_grad_(True)
            self.z_encoder.detune_embedding.weight.requires_grad_(True)
        else:
            self.z_encoder.embedding.weight.requires_grad_(True)
            for module in (
                self.note_release,
                self.context_network,
                self.monophonic_network,
                self.reverb_model,
            ):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = True

        self.set_detune_enabled(stage != "controls")
        
    def alternate_training(self, first_phase=True):
        """Toggle trainability of submodules for the 1st or 2nd training phase.
        Args:
            - first_phase (bool): whether using the 1st phase training strategy
        """
        self.configure_training_stage("controls" if first_phase else "pitch")

    def synthesize_harmonic_part(self, harmonic_synthesizer, amplitudes, harmonic_distribution, inharm_coef, f0_hz):
        params = harmonic_synthesizer.get_controls(amplitudes, harmonic_distribution, inharm_coef, f0_hz)
        harmonic_signal = harmonic_synthesizer(params["amplitudes"],
                                               params["harmonic_distribution"],
                                               params["harmonic_shifts"],
                                               params["f0_hz"])
        return harmonic_signal

    def _synthesize_voices_serial(
        self,
        amplitudes_all,
        harmonics_all,
        inharm_all,
        f0_all,
        magnitudes_all,
    ):
        """Reference implementation that synthesizes one polyphony slot at a time."""
        signal = None
        for voice in range(self.n_synths):
            harmonic = self.synthesize_harmonic_part(
                self.harmonic_synthesizer,
                amplitudes_all[voice],
                harmonics_all[voice],
                inharm_all[voice],
                f0_all[voice],
            )
            noise = self.noise_synthesizer(harmonic, magnitudes_all[voice])
            voice_signal = harmonic + noise
            signal = voice_signal if signal is None else signal + voice_signal
        return signal

    def _synthesize_voices_vectorized(
        self,
        amplitudes_all,
        harmonics_all,
        inharm_all,
        f0_all,
        magnitudes_all,
    ):
        """Merge polyphony and batch axes so the DSP bank runs in one call."""
        voices, batch = amplitudes_all.shape[:2]

        def merge(value):
            return value.reshape(voices * batch, *value.shape[2:])

        harmonic = self.synthesize_harmonic_part(
            self.harmonic_synthesizer,
            merge(amplitudes_all),
            merge(harmonics_all),
            merge(inharm_all),
            merge(f0_all),
        )
        noise = self.noise_synthesizer(harmonic, merge(magnitudes_all))
        return (harmonic + noise).reshape(voices, batch, -1).sum(dim=0)

    def synthesize_voices(
        self,
        amplitudes_all,
        harmonics_all,
        inharm_all,
        f0_all,
        magnitudes_all,
    ):
        if self.synthesis_layout == "vectorized":
            return self._synthesize_voices_vectorized(
                amplitudes_all,
                harmonics_all,
                inharm_all,
                f0_all,
                magnitudes_all,
            )
        return self._synthesize_voices_serial(
            amplitudes_all,
            harmonics_all,
            inharm_all,
            f0_all,
            magnitudes_all,
        )

    def predict_controls(self, conditioning, pedal, piano_model):
        """Return fixed-shape neural controls before non-exportable DSP synthesis."""
        z, global_inharm, global_detuning = self.z_encoder(piano_model)
        context = self.context_network(conditioning, pedal, z)
        reverb_ir = self.reverb_model(piano_model.unsqueeze(-1))

        conditioning, context, global_inharm, global_detuning = self.parallelizer(
            conditioning, context, global_inharm, global_detuning
        )
        extended_pitch = self.note_release(conditioning)
        inharm_coef = self.inharm_model(extended_pitch, global_inharm)
        f0_hz = self.detuner(extended_pitch, global_detuning)
        amplitudes, harmonic_distribution, magnitudes = self.monophonic_network(
            conditioning, extended_pitch, context
        )

        return (
            self.parallelizer.unparallelize_feature(amplitudes),
            self.parallelizer.unparallelize_feature(harmonic_distribution),
            self.parallelizer.unparallelize_feature(inharm_coef),
            self.parallelizer.unparallelize_feature(f0_hz),
            self.parallelizer.unparallelize_feature(magnitudes),
            reverb_ir,
        )

    def predict_controls_stateful(
        self,
        conditioning,
        pedal,
        piano_model,
        extended_pitch,
        context_state,
        monophonic_state,
    ):
        """Predict one or more control frames while carrying explicit GRU state.

        ``extended_pitch`` is maintained by the deployment host so the ONNX graph
        does not unroll the release-state loop for every exported frame.
        """
        z, global_inharm, global_detuning = self.z_encoder(piano_model)
        context, next_context_state = self.context_network.forward_stateful(
            conditioning, pedal, z, context_state
        )
        reverb_ir = self.reverb_model(piano_model.unsqueeze(-1))

        conditioning, context, global_inharm, global_detuning = self.parallelizer(
            conditioning, context, global_inharm, global_detuning
        )
        extended_pitch = self.parallelizer.parallelize_series_operation(extended_pitch)
        inharm_coef = self.inharm_model(extended_pitch, global_inharm)
        f0_hz = self.detuner(extended_pitch, global_detuning)
        amplitudes, harmonic_distribution, magnitudes, next_monophonic_state = (
            self.monophonic_network.forward_stateful(
                conditioning, extended_pitch, context, monophonic_state
            )
        )

        return (
            self.parallelizer.unparallelize_feature(amplitudes),
            self.parallelizer.unparallelize_feature(harmonic_distribution),
            self.parallelizer.unparallelize_feature(inharm_coef),
            self.parallelizer.unparallelize_feature(f0_hz),
            self.parallelizer.unparallelize_feature(magnitudes),
            reverb_ir,
            next_context_state,
            next_monophonic_state,
        )

    def forward(
        self,
        conditioning,
        pedal,
        piano_model):

        amplitudes_all, harmonics_all, inharm_all, f0_all, magnitudes_all, reverb_ir = self.predict_controls(
            conditioning, pedal, piano_model
        )

        signal = self.synthesize_voices(
            amplitudes_all,
            harmonics_all,
            inharm_all,
            f0_all,
            magnitudes_all,
        )
        
        # Keep the dry branch connected to the graph so the training objective
        # can explicitly protect the oscillator quality from an over-large IR.
        non_ir_signal = signal
        signal = self.reverb_module(signal, reverb_ir)
        return signal, reverb_ir, non_ir_signal

            
