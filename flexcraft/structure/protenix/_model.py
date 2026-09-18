import jax
import jax.numpy as jnp

import equinox as eqx
from protenix.backend import load_model
from protenix.protenij import InitialEmbedding, TrunkEmbedding, ConfidenceMetrics
from protenix.protenij import Protenix as _Protenix

from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixResult, ProtenixPrediction

class Protenix:
    def __init__(self, model="protenix_base_default_v1.0.0"):
        self.model = load_model(model)

    def predictor(self, num_recycle=2, num_samples=1,
                  num_sampling_steps=25, deterministic=False):
        jit_predict = eqx.filter_jit(ProtenixEvaluator(
            model=self.model,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)._predict)
        def _predict(key, joltz_spec: ProtenixSpec):
            input, writer = joltz_spec.to_input(pad=True, cache=self.cache)
            # writer_features = {
            #     k: torch.tensor(np.array(v))
            #     for k, v in features.items() if k != "record"
            # }
            # writer_features["record"] = writer_spec["features_dict"]["record"]
            # writer_spec["features_dict"] = writer_features
            #features = jax.tree.map(jnp.array, features)
            prediction = jit_predict(key, input.features, num_samples=num_samples)
            return ProtenixPrediction(
                data=prediction.data,
                writer=writer)
        return _predict

    def evaluator(self, sampling_steps=25, sample_parallel=False,
                 num_recycle=4, stop_recycle_gradient=True):
        evaluator = ProtenixEvaluator(
            model=self.model, sampling_steps=sampling_steps,
            sample_parallel=sample_parallel,
            num_recycle=num_recycle,
            stop_recycle_gradient=stop_recycle_gradient)
        evaluator = eqx.tree_at(
            lambda m: (m.model.gamma0, m.model.step_scale_eta, m.model.noise_scale_lambda, m.model.N_steps),
            evaluator, 
            (0.0, 1.0, 1.0, sampling_steps))
        # return evaluator
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params

class ProtenixEvaluator(eqx.Module):
    model: _Protenix
    sampling_steps: int = 25
    num_recycle: int = 4
    sample_parallel: bool = False
    stop_recycle_gradient: bool = True

    def _predict(self, key, features, recycling_state: TrunkEmbedding | None = None, num_samples=4):
        keys = jax.random.split(key, 10)
        # run embedding
        embedding = self.embedding(keys[0], features)
        state = recycling_state
        if state is None:
            state = TrunkEmbedding(
                s=jnp.zeros_like(embedding.s_init),
                z=jnp.zeros_like(embedding.z_init))
        # run trunk
        state: TrunkEmbedding = self.trunk(keys[1], state, features, embedding)
        # run distogram
        log_distogram: jax.Array = self.model.distogram_head(state.z)
        # run diffusion
        (coords, confidence) = self.structure_module(
            keys[2], state, features, embedding,
            num_samples=num_samples,
            sampling_steps=self.sampling_steps,
            parallel=self.sample_parallel)
        return ProtenixResult(dict(
            features=features, embedding=embedding, state=state,
            distogram=log_distogram, samples=coords, confidence=confidence
        ))
    
    def predict(self, key, protenix_input: ProtenixInput,
                recycling_state: TrunkEmbedding | None = None,
                num_samples=4):
        features = protenix_input.features
        return self._predict(
            key, features, recycling_state=recycling_state,
            num_samples=num_samples)
        

    def embedding(self, key, features) -> InitialEmbedding:
        embedding = self.model.embed_inputs(input_feature_dict=features)
        return embedding

    def trunk(self, key, state: TrunkEmbedding, features, embedding: InitialEmbedding) -> TrunkEmbedding:
        def body(i, trunk_state):
            state, key = trunk_state
            state: TrunkEmbedding
            state = jax.tree.map(jax.lax.stop_gradient, state)
            s, z = state.s, state.z
            z = embedding.z_init + self.model.linear_no_bias_z_cycle(
                self.model.layernorm_z_cycle(z)
            )
            if self.model.template_embedder.n_blocks > 0:
                z = z + self.model.template_embedder(features, z, pair_mask=None, key=key)
            z = self.model.msa_module(
                features,
                z,
                embedding.s_inputs,
                pair_mask=None,
                key=key,
            )
            s = embedding.s_init + self.model.linear_no_bias_s(self.model.layernorm_s(s))
            s, z = self.model.pairformer_stack(
                s, z, pair_mask=None, key=jax.random.fold_in(key, 1)
            )
            return (TrunkEmbedding(s=s, z=z), jax.random.fold_in(key, 1))
        state, key = jax.lax.fori_loop(0, self.num_recycle - 1, body, (state, key))
        if self.stop_recycle_gradient:
            state = jax.lax.stop_gradient(state)
        return body(0, (state, key))[0]

    def structure_module(self, key, state, features, embedding,
                         num_samples=1, sampling_steps=25, parallel=False):
        def single_sample(carry, key):
            sample: jax.Array = self.model.sample_structures(
                initial_embedding=embedding,
                trunk_embedding=state,
                input_feature_dict=features,
                N_samples=1,
                N_steps=sampling_steps,
                key=key,
            )

            confidence: ConfidenceMetrics = self.model.confidence_metrics(
                initial_embedding=embedding,
                trunk_embedding=state,
                input_feature_dict=features,
                coordinates=sample,
                key=key,
            )
            return carry, (sample, confidence)

        if num_samples > 1:
            if parallel:
                vmap_sample = jax.vmap(single_sample, in_axes=(None, 0), out_axes=(None, (0, 0)))
                _, (samples, confidence) = vmap_sample(None, jax.random.split(key, num_samples))
            else:
                _, (samples, confidence) = jax.lax.scan(single_sample, None, jax.random.split(key, num_samples))
        else:
            _, (samples, confidence) = single_sample(None, key)
        return samples, confidence

    # adapted from escalante-bio/protenij under Apache 2.0 license, see NOTICE
    # we need to override this, because of numerics that blows up on CPU but not on GPU
    def sample_structures(self, *,
                          initial_embedding: InitialEmbedding, trunk_embedding: TrunkEmbedding, 
                          input_feature_dict, N_samples, N_steps, key):
        noise_schedule = self.model.inference_noise_scheduler(N_step=N_steps)
        coordinates = sample_diffusion(
            denoise_net=self.model.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=initial_embedding.s_inputs,
            s_trunk=trunk_embedding.s,
            z_trunk=trunk_embedding.z,
            N_sample=N_samples,
            noise_schedule=noise_schedule,
            gamma0=self.model.gamma0,
            gamma_min=self.model.gamma_min,
            noise_scale_lambda=self.model.noise_scale_lambda,
            step_scale_eta=self.model.step_scale_eta,
            key=key,
        )
        return coordinates

# adapted from escalante-bio/protenij under Apache 2.0 license, see NOTICE
# we need to override this, because of numerics in sqrt that blows up on CPU but not on GPU
def sample_diffusion(
    *,
    denoise_net,
    input_feature_dict: dict[str],
    s_inputs,
    s_trunk,
    z_trunk,
    noise_schedule,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    key,
):
    N_atom = input_feature_dict["atom_to_token_idx"].shape[-1]
    batch_shape = s_inputs.shape[:-2]


    # init noise
    # [..., N_sample, N_atom, 3]
    x_l = noise_schedule[0] * jax.random.normal(
        key=key, shape=(*batch_shape, N_sample, N_atom, 3)
    )

    @jax.checkpoint
    def body_function(T, in_T):
        x_l, key = T
        c_tau_last, c_tau = in_T
        x_l = x_l - jnp.mean(x_l, axis=-2, keepdims=True)  # Center the coordinates

        # Denoise with a predictor-corrector sampler
        # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
        gamma = jax.lax.select(c_tau > gamma_min, gamma0, 0.0)
        t_hat = c_tau_last * (gamma + 1)

        # NOTE: this is the culprit that makes all our stuff blow up on CPU
        delta_noise_level = jnp.sqrt(jnp.maximum(t_hat**2 - c_tau_last**2, 0))
        key = jax.random.fold_in(key, 1)
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * jax.random.normal(
            key=key, shape=x_l.shape
        )

        # 2. Denoise from x_{t_hat} to x_{c_tau}
        # Euler step only
        t_hat = (
            jnp.tile(
                t_hat.reshape((1,) * (len(batch_shape) + 1)), (*batch_shape, N_sample)
            )  # [..., N_sample]
        )


        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )

        delta = (x_noisy - x_denoised) / t_hat[
            ..., None, None
        ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
        return (x_l, key), None

    x_l, key = jax.lax.scan(body_function,
        init=(x_l, key),
        xs=(noise_schedule[:-1], noise_schedule[1:]),
    )[0]

    return x_l