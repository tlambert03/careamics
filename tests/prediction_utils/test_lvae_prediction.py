import numpy as np
import pytest
import torch

from careamics.config.inference_model import InferenceConfig
from careamics.lightning import PredictDataModule
from careamics.models.lvae.likelihoods import GaussianLikelihood
from careamics.models.lvae.lvae import LadderVAE
from careamics.prediction_utils import convert_outputs
from careamics.prediction_utils.lvae_prediction import (
    lvae_predict_mmse,
    lvae_predict_single_sample,
)


@pytest.fixture
def minimum_lvae_params():
    return {
        "input_shape": 64,
        "output_channels": 2,
        "multiscale_count": None,
        "z_dims": [128, 128, 128, 128],
        "encoder_n_filters": 64,
        "decoder_n_filters": 64,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        "nonlinearity": "ELU",
        "predict_logvar": "pixelwise",
        "enable_noise_model": False,
        "analytical_kl": False,
    }


@pytest.fixture
def gaussian_likelihood_params():
    return {"predict_logvar": "pixelwise", "logvar_lowerbound": -5}


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_smoke_lvae_prediction(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar
    input_shape = minimum_lvae_params["input_shape"]

    # --- create inference config
    inference_dict = {
        "data_type": "array",
        "axes": "SYX",
        "tile_size": [input_shape, input_shape],
        "tile_overlap": [(input_shape // 4) * 2, (input_shape // 4) * 2],  # ensure even
        "image_means": [2.0],
        "image_stds": [1.0],
        "tta_transforms": False,
    }
    inference_config = InferenceConfig(**inference_dict)

    # --- create dummy dataset
    N_samples = 3
    data_shape = (N_samples, input_shape * 4 + 23, input_shape * 4 + 23)
    data = np.random.random_sample(size=data_shape)
    data_module = PredictDataModule(inference_config, pred_data=data)
    data_module.setup()  # simulate being called by lightning predict loop

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    tiled_predictions = []
    log_vars = []
    for batch in data_module.predict_dataloader():
        y, log_var = lvae_predict_single_sample(model, likelihood_obj, batch)
        tiled_predictions.append(y)
        log_vars.append(log_var)

        # y is a 2-tuple, second element is tile info, similar for logvar
        assert y[0].shape == (1, output_channels, input_shape, input_shape)
        if predict_logvar == "pixelwise":
            assert log_var[0].shape == (1, output_channels, input_shape, input_shape)
        elif predict_logvar is None:
            assert log_var is None

    prediction_shape = (1, output_channels, *data_shape[-2:])
    predictions = convert_outputs(tiled_predictions, tiled=True)
    for prediction in predictions:
        assert prediction.shape == prediction_shape


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_single_sample(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test predictions of a single sample."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    # prediction
    y, log_var = lvae_predict_single_sample(model, likelihood_obj, x)

    assert y.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var is None


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_mmse(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test MMSE prediction."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    # prediction
    y, log_var = lvae_predict_mmse(model, likelihood_obj, x, mmse_count=5)

    assert y.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var is None
