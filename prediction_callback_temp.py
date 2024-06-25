from pathlib import Path

import numpy as np
import tifffile
from pytorch_lightning import Callback

from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.dataset import PathIterableDataset
from careamics.prediction_utils import convert_outputs, create_pred_datamodule

# class CustomPredictAfterValidationCallback(Callback):
#     def __init__(self, pred_datamodule):
#         self.pred_datamodule = pred_datamodule

#     def setup(self, trainer, pl_module, stage):
#         if stage in ("fit", "validate"):
#             # setup the predict data for fit/validate, as we will call it during `on_validation_epoch_end`
#             # not sure if needed, but doesn't hurt until I get it to work
#             self.pred_datamodule.prepare_data()
#             self.pred_datamodule.setup("predict")

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if trainer.sanity_checking:  # optional skip
#             return

#         predictions = trainer.predict(model=pl_module, datamodule=self.pred_datamodule)
#         predictions = convert_outputs(predictions, self.pred_datamodule.tiled)
#         print("Predicted during training.")


class CustomPredictAfterValidationCallback(Callback):
    def __init__(self, pred_datamodule):
        self.pred_datamodule = pred_datamodule

    def setup(self, trainer, pl_module, stage):
        if stage in ("fit", "validate"):
            # setup the predict data for fit/validate, as we will call it during `on_validation_epoch_end`
            # not sure if needed, but doesn't hurt until I get it to work
            self.pred_datamodule.prepare_data()
            self.pred_datamodule.setup("predict")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # optional skip
            return

        # not entirely sure about how preds are returned (and how they must be concatenated), take as pseudocode
        predictions = []
        for idx, batch in enumerate(self.pred_datamodule.predict_dataloader()):
            batch = pl_module._apply_batch_transfer_handler(batch)
            preds = pl_module.predict_step(batch, idx)  # breaks here
            predictions += preds

        predictions = convert_outputs(predictions, self.pred_datamodule.tiled)
        print("Predicted during training.")


n_samples = 2
save_paths = [Path(f"image_{i}.tiff") for i in range(2)]

train_array = np.random.rand(32, 32)
for save_path in save_paths:
    tifffile.imwrite(save_path, train_array)

# print(tifffile.imread(save_path).shape)

config = create_n2v_configuration(
    "PredCallbackTest",
    data_type="tiff",
    axes="YX",
    patch_size=(16, 16),
    batch_size=2,
    num_epochs=3,
    use_augmentations=True,
)
train_dataset = PathIterableDataset(config.data_config, save_paths)

pred_datamodule = create_pred_datamodule(source=save_paths[0], config=config)

predict_after_val_callback = CustomPredictAfterValidationCallback(
    pred_datamodule=pred_datamodule
)
engine = CAREamist(config, callbacks=[predict_after_val_callback])
engine.train(train_source=save_path)
