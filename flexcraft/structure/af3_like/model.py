import equinox as eqx

from flexcraft.structure.common._data import InputSpec

from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixResult, ProtenixPrediction
from flexcraft.structure.protenix._model import Protenix

from flexcraft.structure.boltz._model import Joltz2
from flexcraft.structure.boltz._data import JoltzSpec, JoltzInput
from flexcraft.structure.boltz._result import JoltzResult

from flexcraft.structure.af import AF2Spec, AFWrapper, AFResult, AFInput

CONCRETE_SPEC = dict(
    joltz=JoltzSpec,
    protenix=ProtenixSpec,
    af2=AF2Spec
)

AF3LikeResult = ProtenixResult | JoltzResult | AFResult
AF3LikeInput = ProtenixInput | JoltzInput | AFInput

def _any_spec_to_model_spec(spec: InputSpec, kind):
    return CONCRETE_SPEC[kind](
        *spec.chains,
        templates=spec.templates,
        constraints=spec.constraints
    )

def _get_model_kind(model: str):
    model_kind = None
    if model == "joltz":
        model_kind = "joltz"
    elif model.startswith("protenix"):
        model_kind = "protenix"
    elif model.startswith("af2"):
        model_kind = "af2"
    return model_kind

class AF3LikeSpec(InputSpec):
    def to_input(self, model="joltz", **kwargs):
        model_kind = _get_model_kind(model)
        return _any_spec_to_model_spec(self, model_kind).to_input(**kwargs)

class AF3Like:
    def __init__(self, model="protenix_base_default_v1.0.0", **kwargs):
        """Model-agnostic AF3-like model wrapper.
        
        Args:
            model (str): string model name (protenix_*, joltz, af2_*).
            kwargs: model keyword arguments, e.g. cache dir.
        """
        self.base = None
        self.model_kind = None
        if model == "joltz":
            self.model_kind = "joltz"
            self.base = Joltz2(**kwargs)
        elif model.startswith("protenix"):
            self.model_kind = "protenix"
            self.base = Protenix(model = model)
        elif model.startswith("af2"):
            self.model_kind = "af2"
            self.base = AFWrapper(model = model)

    def predictor(self, **kwargs):
        _predictor = self.base.predictor(**kwargs)
        def _wrapper(key, any_spec: InputSpec):
            spec = _any_spec_to_model_spec(any_spec, self.model_kind)
            return _predictor(key, spec)
        return _wrapper

    def evaluator(self, **kwargs):
        return self.base.evaluator(**kwargs)
