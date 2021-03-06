# 'air density'
air_density_trans = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        logits=t[..., 9:12]
    ),
    components_distribution=tfd.Normal(
        loc=t[..., 12:15],
        scale=tfb.Softplus().forward(t[..., 15:18])
    ),
    name=f'learnable_MixtureNormal_obj_1'
)
# 'wind direction'
wind_direction = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        logits=t[..., 18:21]
    ),
    components_distribution=tfd.VonMises(
        loc=t[..., 21:24],
        concentration=tfb.Softplus().forward(t[..., 24:27])
    ),
    name=f'learnable_MixtureVonMises_obj'
)