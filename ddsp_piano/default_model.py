from ddsp_piano.modules import sub_modules
from ddsp_piano.modules.piano_model import PianoModel

from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.ddsp_pytorch.reverb import Reverb
from ddsp_piano.ddsp_pytorch.fdn import FDNReverb
from ddsp_piano.modules.v2_sub_modules import (
    FiLMContextNetwork,
    JointParametricInharmTuning,
    MonophonicDeepNetwork,
    V2FDNReverbControls,
)


def build_polyphonic_ddsp_module(
    sample_rate=16000,
    duration=3,
    inference=False,
    reverb_wet_gain=0.25):
    """ Polyphonic bank of additive + filtered noise synthesizers.
    Args:
        - sample_rate (int): number of samples per second.
        - duration (float): length of generated sample (in seconds).
        - inference (bool): synthesis for inference (slower but can handle
        longer sequences).
    Returns:
        - ddsp_module 
    """
    n_samples = int(duration * sample_rate)

    # Init Harmonic + Noise Synthesizers
    harmonic_synthesizer = MultiInharmonic(n_samples, sample_rate, inference=inference)
    noise_synthesizer = Noise()
    reverb_effects = Reverb(wet_gain=reverb_wet_gain)

    return harmonic_synthesizer, noise_synthesizer, reverb_effects

def get_model(
    inference=False,
    duration=3,
    n_synths=16,
    n_substrings=2,
    n_piano_models=10,
    piano_embedding_dim=16,
    n_noise_filter_banks=64,
    frame_rate=250,
    sample_rate=16000,
    reverb_duration=1.5,
    reverb_wet_gain=0.25):
    # Self-contained sub-modules
    z_encoder = sub_modules.OneHotZEncoder(n_instruments=n_piano_models, z_dim=piano_embedding_dim, n_frames=int(duration * frame_rate))
    note_release = sub_modules.NoteRelease(frame_rate=frame_rate)
    parallelizer = sub_modules.Parallelizer(n_synths=n_synths)
    inharm_model = sub_modules.InharmonicityNetwork()
    detuner = sub_modules.Detuner(n_substrings=n_substrings)
    reverb_model = sub_modules.MultiInstrumentReverb(
        n_instruments=n_piano_models,
        reverb_length=int(reverb_duration * sample_rate),
        inference=inference,
        apply_decay=True,
    )

    # Neural modules
    context_network = sub_modules.ContextNetwork()

    monophonic_network = sub_modules.MonophonicNetwork()

    harmonic_synthesizer, noise_synthesizer, reverb_module = build_polyphonic_ddsp_module(
        sample_rate=sample_rate,
        duration=duration,
        inference=inference,
        reverb_wet_gain=reverb_wet_gain,
    )

    # Full piano model definition
    model = PianoModel(
        n_synths=n_synths,
        z_encoder=z_encoder,
        note_release=note_release,
        context_network=context_network,
        parallelizer=parallelizer,
        monophonic_network=monophonic_network,
        inharm_model=inharm_model,
        detuner=detuner,
        reverb_model=reverb_model,
        harmonic_synthesizer=harmonic_synthesizer,
        noise_synthesizer=noise_synthesizer,
        reverb_module=reverb_module
    )

    return model


def get_v2_model(
    inference=False,
    duration=3,
    n_synths=16,
    n_piano_models=10,
    piano_embedding_dim=16,
    frame_rate=250,
    sample_rate=16000,
    reverb_duration=1.5,
):
    """Build the independent PyTorch DDSP-Piano v2-style model.

    The neural graph predicts 128 harmonic bins, 96 noise bands, and nine FDN
    controls. FDN synthesis remains in ``ddsp_piano.ddsp_pytorch.fdn`` so the
    ONNX contract contains only conservative control-network operators.
    """
    z_encoder = sub_modules.OneHotZEncoder(
        n_instruments=n_piano_models,
        z_dim=piano_embedding_dim,
        n_frames=int(duration * frame_rate),
    )
    model = PianoModel(
        n_synths=n_synths,
        z_encoder=z_encoder,
        note_release=sub_modules.NoteRelease(frame_rate=frame_rate),
        context_network=FiLMContextNetwork(z_dim=piano_embedding_dim),
        parallelizer=sub_modules.Parallelizer(n_synths=n_synths),
        monophonic_network=MonophonicDeepNetwork(),
        inharm_model=JointParametricInharmTuning(),
        detuner=sub_modules.Detuner(n_substrings=2),
        reverb_model=V2FDNReverbControls(n_piano_models),
        harmonic_synthesizer=MultiInharmonic(
            int(duration * sample_rate), sample_rate, inference=inference
        ),
        noise_synthesizer=Noise(),
        reverb_module=FDNReverb(
            sample_rate=sample_rate,
            length=int(reverb_duration * sample_rate),
        ),
    )
    return model
