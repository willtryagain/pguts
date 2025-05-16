from typing import Callable, List, Mapping, Optional, Type, Union

import torch
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor

from ..utils import k_hop_subgraph_sampler


class FCSPINImputer(Imputer):
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,
        loss_fn: Callable,
        scale_target: bool = True,
        whiten_prob: Union[float, List[float]] = 0.2,
        n_roots_subgraph: Optional[int] = None,
        n_hops: int = 2,
        max_edges_subgraph: Optional[int] = 1000,
        cut_edges_uniformly: bool = False,
        prediction_loss_weight: float = 1.0,
        metrics: Optional[Mapping[str, Metric]] = None,
        scheduler_class: Optional = None,
        scheduler_kwargs: Optional[Mapping] = None,
        scaler=None,
    ):
        super().__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            whiten_prob=whiten_prob,
            prediction_loss_weight=prediction_loss_weight,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.n_roots = n_roots_subgraph
        self.n_hops = n_hops
        self.max_edges_subgraph = max_edges_subgraph
        self.cut_edges_uniformly = cut_edges_uniformly
        self.scaler = scaler

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training and self.n_roots is not None:
            batch = k_hop_subgraph_sampler(
                batch,
                self.n_hops,
                self.n_roots,
                max_edges=self.max_edges_subgraph,
                cut_edges_uniformly=self.cut_edges_uniformly,
            )
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def training_step(self, batch, batch_idx):
        device = batch.y.device
        batch.y = self.scaler.transform(batch.y.cpu())
        batch.y = batch.y.to(device)
        eval_mask = torch.zeros_like(batch.eval_mask).bool()
        eval_mask[:, -12:] = True

        y_hat, y, loss = self.shared_step(batch, mask=eval_mask)

        # Logging

        self.train_metrics.update(y_hat, y, eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss("train", loss, batch_size=batch.batch_size)

        if "target_nodes" in batch:
            torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        # batch.input.target_mask = batch.eval_mask
        device = batch.y.device
        batch.y = self.scaler.transform(batch.y.cpu())
        batch.y = batch.y.to(device)
        eval_mask = torch.zeros_like(batch.eval_mask).bool()
        eval_mask[:, -12:] = True
        y_hat, y, val_loss = self.shared_step(batch, eval_mask)

        # Logging
        self.val_metrics.update(y_hat, y, eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss("val", val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):

        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        y_og = batch.y.clone()
        device = batch.y.device
        batch.y = self.scaler.transform(batch.y.cpu())
        batch.y = batch.y.to(device)
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y_hat = self.scaler.inverse_transform(y_hat.cpu())
        y_hat = y_hat.to(device)

        y, eval_mask = y_og, batch.eval_mask
        test_loss = self.loss_fn(y_hat, y, eval_mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss("test", test_loss, batch_size=batch.batch_size)
        return test_loss

    def predict_batch(
        self,
        batch,
        preprocess=False,
        postprocess=True,
        return_target=False,
        forward_kwargs=None,
    ):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :param forward_kwargs: optional, additional keyword arguments passed to the forward method.
        :return: (y_true), y_hat, (mask)
        """
        inputs, targets, mask, transform = self._unpack_batch(batch)
        mask = torch.ones_like(mask).bool()
        mask[:, -12:] = False # 50% missing
        inputs["mask"] = mask
        inputs["x"][~mask] = 0

        batch.eval_mask = ~mask

        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        y_hat = self.forward(**inputs, **forward_kwargs)
        # Rescale outputs
        if postprocess:
            trans = transform.get("y")
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get("y")
            return y, y_hat, mask
        return y_hat

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y_og = batch.y.clone()
        device = batch.y.device
        batch.y = self.scaler.transform(batch.y.cpu())
        batch.y = batch.y.to(device)
        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)
        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]
        y_hat = self.scaler.inverse_transform(y_hat.cpu())
        y_hat = y_hat.to(device)
        y_hat_reset = torch.where(batch.mask.bool(), y_og, y_hat)
        output = dict(
            y=y_og, y_hat=y_hat, y_hat_reset=y_hat_reset, mask=batch.eval_mask
        )
        return output

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument("--whiten-prob", type=float, default=0.05)
        parser.add_argument("--prediction-loss-weight", type=float, default=1.0)
        parser.add_argument("--n-roots-subgraph", type=int, default=None)
        parser.add_argument("--n-hops", type=int, default=2)
        parser.add_argument("--max-edges-subgraph", type=int, default=1000)
        parser.add_argument("--cut-edges-uniformly", type=bool, default=False)
        return parser
