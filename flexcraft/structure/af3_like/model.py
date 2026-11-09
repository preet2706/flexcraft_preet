import inspect
import warnings

import equinox as eqx

from flexcraft.structure.common._data import InputSpec

from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixResult, ProtenixPrediction
from flexcraft.structure.protenix._model import Protenix

from flexcraft.structure.boltz._model import Joltz2
from flexcraft.structure.boltz._data import JoltzSpec, JoltzInput
from flexcraft.structure.boltz._result import JoltzResult

from flexcraft.structure.af import AF2Spec, AFWrapper, AFResult, AFInput

# NOTE: no `import *` of this module's AF3LikeResult: flexcraft.structure.common._result
# defines a class of the same name (the Boltz/Protenix result base class) that pipelines import.
__all__ = ["AF3Like", "AF3LikeSpec", "CONCRETE_SPEC", "MODEL_ALIASES", "resolve_model",
           "model_kind", "uses_msa", "supports_samples", "template_chains_are_complex_chains",
           "to_input_kwargs", "model_init_kwargs"]

CONCRETE_SPEC = dict(
    joltz=JoltzSpec,
    protenix=ProtenixSpec,
    af2=AF2Spec
)

_BASE_MODEL = dict(
    joltz=Joltz2,
    protenix=Protenix,
    af2=AFWrapper
)

AF3LikeResult = ProtenixResult | JoltzResult | AFResult
AF3LikeInput = ProtenixInput | JoltzInput | AFInput

def _any_spec_to_model_spec(spec: InputSpec, kind):
    if kind is None:
        raise ValueError("Cannot convert a spec for an unknown model kind.")
    if type(spec) is CONCRETE_SPEC[kind]:
        return spec
    result = CONCRETE_SPEC[kind](
        *spec.chains,
        templates=spec.templates,
        constraints=spec.constraints
    )
    # Carry over what the chains / templates / constraints triple does not cover.
    # precomputed_msas: dropping it silently turns Boltz complex predictions single-sequence.
    # temporaries: owns the lifetime of temporary template files written by add_template.
    result.precomputed_msas = list(spec.precomputed_msas)
    result.temporaries = list(spec.temporaries)
    return result

def _get_model_kind(model: str):
    model_kind = None
    if model == "joltz":
        model_kind = "joltz"
    elif model.startswith("protenix"):
        model_kind = "protenix"
    elif model.startswith("af2"):
        model_kind = "af2"
    return model_kind

# --- user-facing model names (e.g. for a --structure_model flag) ---------------------------
# Aliases cover the names the pipelines used before AF3Like existed; any other name is passed
# through to AF3Like unchanged (e.g. "protenix-v2", "af2_model_3_ptm").
MODEL_ALIASES = dict(
    boltz2="joltz",
    boltz="joltz",
    protenix_v1="protenix_base_default_v1.0.0",
    protenix_v2="protenix-v2",
    af2="af2_model_1_ptm",
)

def resolve_model(model: str) -> str:
    """AF3Like model name for a user-facing model name or alias."""
    return MODEL_ALIASES.get(model, model)

def model_kind(model: str) -> str:
    """'joltz', 'protenix' or 'af2' for a model name or alias. Raises on unknown names."""
    kind = _get_model_kind(resolve_model(model))
    if kind is None:
        raise ValueError(
            f"unknown model '{model}'. Use one of the aliases {sorted(MODEL_ALIASES)}, "
            f"'joltz', or a name starting with 'protenix' or 'af2'.")
    return kind

def uses_msa(model: str) -> bool:
    """Whether the model consumes an MSA. AF2Spec.to_input drops MSAs."""
    return model_kind(model) != "af2"

def supports_samples(model: str) -> bool:
    """Whether the model returns several diffusion samples. AF2 returns exactly one."""
    return model_kind(model) != "af2"

def template_chains_are_complex_chains(model: str) -> bool:
    """Meaning of `to_chains` in add_template: Protenix and AF2 scatter the template onto the
    named chains OF THE COMPLEX; Boltz names chains WITHIN the template file."""
    return model_kind(model) != "joltz"

def to_input_kwargs(model: str, pad=True, cache=None) -> dict:
    """kwargs for AF3LikeSpec.to_input(model=...): only JoltzSpec takes pad / cache."""
    if model_kind(model) != "joltz":
        return {}
    kwargs = dict(pad=pad)
    if cache is not None:
        kwargs["cache"] = cache
    return kwargs

def model_init_kwargs(model: str, boltz_cache=None, af_cache=None) -> dict:
    """kwargs for AF3Like(model, ...). Protenix downloads to ~/.protenix, no cache argument."""
    kind = model_kind(model)
    if kind == "joltz" and boltz_cache is not None:
        return dict(cache=boltz_cache)
    if kind == "af2" and af_cache is not None:
        return dict(cache=af_cache)
    return {}

class AF3LikeSpec(InputSpec):
    def to_input(self, model="joltz", **kwargs):
        model_kind = _get_model_kind(resolve_model(model))
        return _any_spec_to_model_spec(self, model_kind).to_input(**kwargs)

class AF3Like:
    def __init__(self, model="protenix_base_default_v1.0.0", **kwargs):
        """Model-agnostic AF3-like model wrapper.
        
        Args:
            model (str): string model name (protenix_*, joltz, af2_*) or alias (see MODEL_ALIASES).
            kwargs: model keyword arguments, e.g. cache dir. Arguments the selected
                model does not take are dropped with a warning.
        """
        model = resolve_model(model)
        self.model = model
        self.model_kind = _get_model_kind(model)
        if self.model_kind is None:
            raise ValueError(
                f"unknown AF3-like model '{model}': expected 'joltz', or a name starting "
                f"with 'protenix' or 'af2'.")
        base = _BASE_MODEL[self.model_kind]
        accepted = set(inspect.signature(base.__init__).parameters) - {"self", "model"}
        ignored = sorted(set(kwargs) - accepted)
        if ignored:
            warnings.warn(f"{base.__name__} ignores arguments {ignored}.")
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        if self.model_kind == "joltz":
            # "joltz" names the kind, not a Joltz2 checkpoint: do not forward it.
            self.base = base(**kwargs)
        else:
            self.base = base(model=model, **kwargs)

    def predictor(self, **kwargs):
        _predictor = self.base.predictor(**kwargs)
        def _wrapper(key, any_spec: InputSpec):
            spec = _any_spec_to_model_spec(any_spec, self.model_kind)
            return _predictor(key, spec)
        return _wrapper

    def evaluator(self, **kwargs):
        """Returns (params -> evaluator, params). Shared kwargs: num_recycle, num_sampling_steps."""
        return self.base.evaluator(**kwargs)
