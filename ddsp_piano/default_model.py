from ddsp_piano.modules import sub_modules
from ddsp_piano.modules.piano_model import PianoModel

from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.ddsp_pytorch.reverb import Reverb
from ddsp_piano.ddsp_pytorch.fdn import FDNReverb
from ddsp_piano.modules.configurable_sub_modules import (
    FiLMContextNetwork,
    JointParametricInharmTuning,
    MonophonicDeepNetwork,
    ResidualDeepMonophonicNetwork,
    ResidualFiLMContextNetwork,
    ResidualJointInharmonicity,
    FDNReverbControls,
)


def build_polyphonic_ddsp_module(
    sample_rate=16000,
    duration=3,
    inference=False,
    reverb_wet_gain=0.25,
    n_harmonics=96):
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
    harmonic_synthesizer = MultiInharmonic(
        n_samples,
        sample_rate,
        inference=inference,
        n_harmonics=n_harmonics,
    )
    noise_synthesizer = Noise()
    reverb_effects = Reverb(wet_gain=reverb_wet_gain)

    return harmonic_synthesizer, noise_synthesizer, reverb_effects

def build_paper_model(
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
    reverb_wet_gain=0.25,
    synthesis_layout="serial"):
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
        n_harmonics=96,
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
        reverb_module=reverb_module,
        synthesis_layout=synthesis_layout,
    )

    return model


def build_configurable_model(
    inference=False,
    duration=3,
    n_synths=16,
    n_piano_models=10,
    piano_embedding_dim=16,
    n_harmonics=128,
    n_noise_filter_banks=96,
    frame_rate=250,
    sample_rate=16000,
    reverb_duration=1.5,
    reverb_type="fdn",
    reverb_wet_gain=0.25,
    context_type="film",
    monophonic_type="deep",
    inharmonicity_type="joint",
    synthesis_layout="serial",
):
    """Build a configurable DDSP-Piano control model.

    Harmonic/noise dimensions and host-side reverb are explicit experiment
    parameters. FFT synthesis remains outside every exported ONNX graph.
    """
    if n_harmonics <= 0 or n_noise_filter_banks <= 0:
        raise ValueError("harmonic and noise dimensions must be positive")
    if reverb_type not in {"ir", "fdn"}:
        raise ValueError("reverb_type must be 'ir' or 'fdn'")
    if context_type not in {"legacy", "residual_film", "film"}:
        raise ValueError("invalid context_type")
    if monophonic_type not in {"legacy", "residual_deep", "deep"}:
        raise ValueError("invalid monophonic_type")
    if inharmonicity_type not in {"legacy", "residual_joint", "joint"}:
        raise ValueError("invalid inharmonicity_type")
    if monophonic_type != "deep" and (
        n_harmonics != 96 or n_noise_filter_banks != 64
    ):
        raise ValueError(
            "legacy and residual monophonic networks require 96 harmonics and 64 noise bands"
        )
    z_encoder = sub_modules.OneHotZEncoder(
        n_instruments=n_piano_models,
        z_dim=piano_embedding_dim,
        n_frames=int(duration * frame_rate),
    )
    if reverb_type == "fdn":
        reverb_model = FDNReverbControls(n_piano_models)
        reverb_module = FDNReverb(
            sample_rate=sample_rate,
            length=int(reverb_duration * sample_rate),
        )
    else:
        reverb_model = sub_modules.MultiInstrumentReverb(
            n_instruments=n_piano_models,
            reverb_length=int(reverb_duration * sample_rate),
            inference=inference,
            apply_decay=True,
        )
        reverb_module = Reverb(wet_gain=reverb_wet_gain)

    context_networks = {
        "legacy": sub_modules.ContextNetwork,
        "residual_film": lambda: ResidualFiLMContextNetwork(z_dim=piano_embedding_dim),
        "film": lambda: FiLMContextNetwork(
            z_dim=piano_embedding_dim,
            n_synths=n_synths,
        ),
    }
    monophonic_networks = {
        "legacy": sub_modules.MonophonicNetwork,
        "residual_deep": ResidualDeepMonophonicNetwork,
        "deep": lambda: MonophonicDeepNetwork(
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_filter_banks,
        ),
    }
    inharmonicity_networks = {
        "legacy": sub_modules.InharmonicityNetwork,
        "residual_joint": ResidualJointInharmonicity,
        "joint": JointParametricInharmTuning,
    }

    model = PianoModel(
        n_synths=n_synths,
        z_encoder=z_encoder,
        note_release=sub_modules.NoteRelease(frame_rate=frame_rate),
        context_network=context_networks[context_type](),
        parallelizer=sub_modules.Parallelizer(n_synths=n_synths),
        monophonic_network=monophonic_networks[monophonic_type](),
        inharm_model=inharmonicity_networks[inharmonicity_type](),
        detuner=sub_modules.Detuner(n_substrings=2),
        reverb_model=reverb_model,
        harmonic_synthesizer=MultiInharmonic(
            int(duration * sample_rate),
            sample_rate,
            inference=inference,
            n_harmonics=n_harmonics,
        ),
        noise_synthesizer=Noise(),
        reverb_module=reverb_module,
        synthesis_layout=synthesis_layout,
    )
    model.reverb_type = reverb_type
    model.architecture = {
        "context_type": context_type,
        "monophonic_type": monophonic_type,
        "inharmonicity_type": inharmonicity_type,
    }
    return model
