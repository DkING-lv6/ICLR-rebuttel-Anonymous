from abc import ABC
from abc import abstractmethod
import time
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import to_undirected
from utils_tae import seed_everything


class BaseGraphAugmenter(ABC):
    """
    Abstract base class for graph data augmentation strategies.

    Methods:
    - init_with_data(self, data)
        Initialize the augmenter with graph data.

    - augment(self, model, x, edge_index)
        Perform graph augmentation.

    - adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor)
        Adapt labels and training mask after augmentation.
    """

    @abstractmethod
    def init_with_data(self, data: pyg.data.Data):
        """
        Initialize the augmenter with graph data.

        Parameters:
        - data: pyg.data.Data
            Graph data used for initialization.
        """
        pass

    @abstractmethod
    def augment(
        self, model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor
    ):
        """
        Perform graph augmentation.

        Parameters:
        - model: torch.nn.Module
            Graph neural network model.
        - x: torch.Tensor
            Input features of the graph nodes.
        - edge_index: torch.Tensor
            Edge indices of the graph.

        Returns:
        - augmented_x: torch.Tensor
            Augmented node features.
        - augmented_edge_index: torch.Tensor
            Augmented edge indices.
        - runtime_info: dict
            Additional runtime information from the augmentation process.
        """
        pass

    @abstractmethod
    def adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor):
        """
        Adapt labels and training mask after augmentation.

        Parameters:
        - y: torch.Tensor
            Original node labels.
        - train_mask: torch.Tensor
            Original training mask.

        Returns:
        - adapted_y: torch.Tensor
            Adapted node labels.
        - adapted_train_mask: torch.Tensor
            Adapted training mask.
        """
        pass


class TAEAugmenter(BaseGraphAugmenter):
    """
    Topological Augmentation Engine (TAE) for graph data.

    Parameters:
    - random_state: int or None, optional (default: None)
        Random seed for reproducibility.

    Methods:
    - __init__(self, random_state: int = None)
        Initializes the TAEAugmenter instance.

    - init_with_data(self, data: pyg.data.Data)
        Initializes the augmenter with graph data.

    - augment(self, model, x, edge_index)
        Performs topology-aware graph augmentation.

    - adapt_labels_and_train_mask(self, y, train_mask)
        Adapts labels and training mask after augmentation.

    - info(self)
        Prints information about the augmenter.

    - predict_proba(model, x, edge_index, return_numpy=False)
        Computes predicted class probabilities using the model.

    - edge_sampling(edge_index, edge_sampling_proba, random_state=None)
        Performs edge sampling based on probability.

    - get_group_mean(values, labels, classes)
        Computes the mean of values within each class.

    - get_virtual_node_features(x, y_pred, classes)
        Computes virtual node features based on predicted labels.

    - get_connectivity_distribution_sparse(y_pred, edge_index, n_class, n_node, n_edge)
        Computes the distribution of neighbor labels for each node.

    - get_node_risk(self, y_pred_proba, y_pred)
        Computes node risk based on predicted class probabilities.

    - estimate_node_posterior_likelihood(self, y_pred_proba, y_neighbor_distr)
        Computes posterior likelihood for each node and class.

    - get_virual_link_proba(self, node_posterior, y_pred)
        Computes virtual link probabilities based on node posterior likelihood.
    """

    def __init__(
        self,
        random_state: int = None,
    ):
        """
        Initializes the TAEAugmenter instance.

        Parameters:
        - random_state: int or None, optional (default: None)
            Random seed for reproducibility.
        """
        super().__init__()
        # parameter check
        assert (
            isinstance(random_state, int) or random_state is None
        ), "random_state must be an integer or None"
        self.random_state = random_state
        self.init_flag = False

    def init_with_data(self, data: pyg.data.Data):
        """
        Initializes the augmenter with graph data.

        Parameters:
        - data: pyg.data.Data
            The graph data.

        Raises:
        - AssertionError: If data is not a pyg.data.Data object or lacks required attributes.

        Returns:
        - self: TAEAugmenter
        """
        assert isinstance(data, pyg.data.Data), "data must be a pyg.data.Data object"
        assert hasattr(data, "train_mask"), "data must have 'train_mask' attribute"
        assert hasattr(data, "val_mask"), "data must have 'val_mask' attribute"
        assert hasattr(data, "test_mask"), "data must have 'test_mask' attribute"

        # initialization
        x, edge_index, train_mask, y_train, device = (
            data.x,
            data.edge_index,
            data.train_mask,
            data.y[data.train_mask],
            data.x.device,
        )
        classes, train_class_counts = y_train.unique(return_counts=True)
        self.classes = classes
        self.train_class_counts = train_class_counts
        self.n_node = x.shape[0]
        self.n_edge = edge_index.shape[1]
        self.n_class = len(classes)
        self.y_virtual = classes
        self.y_train = y_train
        self.train_mask = train_mask
        self.train_class_weights = train_class_counts / train_class_counts.max()
        # raw_weights = train_class_counts.max() / train_class_counts
        # self.train_class_weights = raw_weights / raw_weights.max()
        self.empty_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        self.device = device
        self.init_flag = True

        return self

    def augment(
            self, model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor
    ):
        """
        Performs topology-aware graph augmentation with refined pseudo-labels.

        Parameters:
        - model: torch.nn.Module
            The model used for prediction.
        - x: torch.Tensor
            Node features.
        - edge_index: torch.Tensor
            Edge indices.

        Returns:
        - x_aug: torch.Tensor
            Augmented node features.
        - edge_index_aug: torch.Tensor
            Augmented edge indices.
        - info: dict
            Augmentation information.
        """

        assert self.init_flag, "init_with_data() must be called before augment()"

        if self.random_state is not None:
            self.random_state += 1
        train_mask = self.train_mask

        start_time = time.time()
        y_pred_proba = self.predict_proba(model, x, edge_index)
        y_pred = y_pred_proba.argmax(axis=1)

        y_pred[train_mask] = self.y_train


        node_risk = self.get_node_risk(y_pred_proba, y_pred)
        start_time_sim = time.time()


        y_neighbor_distr = self.get_connectivity_distribution_sparse(
            y_pred, edge_index, self.n_class, self.n_node, self.n_edge
        )


        node_posterior = self.estimate_node_posterior_likelihood(
            y_pred_proba, y_neighbor_distr
        )


        y_pred_refined = self.refine_predictions_with_risk(
            y_pred, node_posterior, node_risk, train_mask, y_pred_proba
        )


        virtual_link_proba = self.get_virual_link_proba(node_posterior, y_pred)
        time_cost_sim = time.time() - start_time_sim

        start_time_gen = time.time()


        virtual_link_proba *= node_risk.reshape(-1, 1)


        virtual_adj = virtual_link_proba.T.to_sparse().coalesce()
        edge_index_candidates, edge_sampling_proba = (
            virtual_adj.indices(),
            virtual_adj.values(),
        )

        virtual_edge_index = self.edge_sampling(
            edge_index_candidates, edge_sampling_proba, self.random_state
        )
        virtual_edge_index[0] += self.n_node
        virtual_edge_index = to_undirected(virtual_edge_index)


        x_virtual = self.get_virtual_node_features(x, y_pred_refined, self.classes)
        time_cost_gen = time.time() - start_time_gen


        time_cost = time.time() - start_time
        x_aug = torch.concat([x, x_virtual])
        edge_index_aug = torch.concat([edge_index, virtual_edge_index], axis=1)
        time_cost = time.time() - start_time

        info = {
            "time_aug(ms)": time_cost * 1000,
            "time_unc(ms)": self.time_unc_comp * 1000,
            "time_risk(ms)": self.time_risk_comp * 1000,
            "time_sim(ms)": time_cost_sim * 1000,
            "time_neighbor_distr(ms)": self.time_neighbor_distr * 1000,
            "time_gen(ms)": time_cost_gen * 1000,
            "node_ratio(%)": x_aug.shape[0] / x.shape[0] * 100,
            "edge_ratio(%)": edge_index_aug.shape[1] / edge_index.shape[1] * 100,
        }
        return x_aug, edge_index_aug, info, y_pred_refined

    def refine_predictions_with_risk(
            self,
            y_pred: torch.Tensor,
            node_posterior: torch.Tensor,
            node_risk: torch.Tensor,
            train_mask: torch.Tensor,
            y_pred_proba: torch.Tensor
    ):

        y_pred_refined = y_pred.clone()

        # high_risk_mask = (node_risk > 0) & (~train_mask)
        high_risk_mask = (node_risk > 0)
        # high_risk_mask = (node_risk > 0) & train_mask & (~confirmed_normal_mask)

        if high_risk_mask.sum() > 0:

            enhanced_predictions = node_posterior[high_risk_mask].argmax(dim=1)


            original_confidence = y_pred_proba[high_risk_mask].max(dim=1).values
            enhanced_confidence = node_posterior[high_risk_mask].max(dim=1).values


            confident_mask = enhanced_confidence > original_confidence
            update_indices = torch.where(high_risk_mask)[0][confident_mask]
            y_pred_refined[update_indices] = enhanced_predictions[confident_mask]

        return y_pred_refined

    def adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor):

        new_y = torch.concat([y, self.y_virtual])
        new_train_mask = torch.concat(
            [train_mask, torch.ones_like(self.y_virtual).bool()]
        )
        return new_y, new_train_mask

    def info(self):
        """
        Prints information about the augmenter.
        """
        print(
            f"TAEAugmenter(\n"
            f"    n_node={self.n_node},\n"
            f"    n_edge={self.n_edge},\n"
            f"    n_class={self.n_class},\n"
            f"    classes={self.classes.cpu()},\n"
            f"    train_class_counts={self.train_class_counts.cpu()},\n"
            f"    train_class_weights={self.train_class_weights.cpu()},\n"
            f"    device={self.device},\n"
            f")"
        )

    @staticmethod
    def predict_proba(model, x, edge_index, return_numpy=False):

        model.eval()
        with torch.no_grad():
            logits = model.forward(x, edge_index)


        if logits.shape[1] == 1:

            anomaly_scores = logits.squeeze(1)
            prob_anomaly = torch.sigmoid(anomaly_scores)
            prob_normal = 1 - prob_anomaly

            pred_proba = torch.stack([prob_normal, prob_anomaly], dim=1)

        else:
            pred_proba = torch.softmax(logits, dim=1)

        pred_proba = pred_proba.detach()
        if return_numpy:
            pred_proba = pred_proba.cpu().numpy()
        return pred_proba

    @staticmethod
    def edge_sampling(
        edge_index: torch.Tensor,
        edge_sampling_proba: torch.Tensor,
        random_state: int = None,
    ):
        """
        Performs edge sampling based on probability.

        Parameters:
        - edge_index: torch.Tensor
            Edge indices.
        - edge_sampling_proba: torch.Tensor
            Edge sampling probabilities.
        - random_state: int or None, optional (default: None)
            Random seed for reproducibility.

        Returns:
        - sampled_edge_index: torch.Tensor
            Sampled edge indices.
        """
        if edge_sampling_proba.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_sampling_proba.device)
        assert edge_sampling_proba.min() >= 0 and edge_sampling_proba.max() <= 1
        seed_everything(random_state)
        edge_sample_mask = torch.rand_like(edge_sampling_proba) < edge_sampling_proba
        return edge_index[:, edge_sample_mask]

    @staticmethod
    def get_group_mean(
        values: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor
    ):
        """
        Computes the mean of values within each class.

        Parameters:
        - values: torch.Tensor
            Values to compute the mean of.
        - labels: torch.Tensor
            Labels corresponding to values.
        - classes: torch.Tensor
            Classes for which to compute the mean.

        Returns:
        - new_values: torch.Tensor
            Mean values for each class.
        """
        new_values = torch.zeros_like(values)
        for i in classes:
            mask = labels == i
            new_values[mask] = values[mask].mean()
        return new_values

    @staticmethod
    def get_virtual_node_features(x: torch.Tensor, y_pred: torch.Tensor, classes: list):
        """
        Computes virtual node features based on predicted labels.

        Parameters:
        - x: torch.Tensor
            Node features.
        - y_pred: torch.Tensor
            Predicted labels.
        - classes: list
            Unique classes in the dataset.

        Returns:
        - virtual_node_features: torch.Tensor
            Virtual node features for each class.
        """
        return torch.stack([x[y_pred == label].mean(axis=0) for label in classes])

    def get_connectivity_distribution_sparse(
        self,
        y_pred: torch.Tensor,
        edge_index: torch.Tensor,
        n_class: int,
        n_node: int,
        n_edge: int,
    ):
        """
        Computes the distribution of connectivity labels.

        Parameters:
        - y_pred: torch.Tensor
            Predicted labels.
        - edge_index: torch.Tensor
            Edge indices (sparse).
        - n_class: int
            Number of classes.
        - n_node: int
            Number of nodes.
        - n_edge: int
            Number of edges.

        Returns:
        - neighbor_y_distr: torch.Tensor
            Normalized connectivity label distribution.
        """
        start_time = time.time()

        device = y_pred.device
        edge_dest_class = torch.zeros(
            (n_edge, n_class), dtype=torch.int, device=device
        ).scatter_(
            1, y_pred[edge_index[1]].unsqueeze(1), 1
        )  # [n_edges, n_class]
        neighbor_y_distr = (
            torch.zeros((n_node, n_class), dtype=torch.int, device=device)
            .scatter_add_(
                dim=0,
                index=edge_index[0].repeat(n_class, 1).T,
                src=edge_dest_class,
            )
            .float()
        )


        neighbor_y_distr /= neighbor_y_distr.sum(axis=1).reshape(-1, 1)
        neighbor_y_distr = neighbor_y_distr.nan_to_num(0)

        self.time_neighbor_distr = time.time() - start_time
        return neighbor_y_distr

    def get_node_risk(self, y_pred_proba: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes node risk based on predicted probabilities.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - node_risk: torch.Tensor
            Node risk scores.
        """

        start_time = time.time()

        node_unc = 1 - y_pred_proba.max(axis=1).values
        self.time_unc_comp = time.time() - start_time

        node_unc_class_mean = self.get_group_mean(node_unc, y_pred, self.classes)

        node_risk = (node_unc - node_unc_class_mean).clip(min=0)

        node_risk *= self.train_class_weights[y_pred]
        self.time_risk_comp = time.time() - start_time
        return node_risk

    def estimate_node_posterior_likelihood(
        self, y_pred_proba: torch.Tensor, y_neighbor_distr: torch.Tensor
    ):
        """
        Estimates node posterior likelihood for each class.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_neighbor_distr: torch.Tensor
            Connectivity label distribution.

        Returns:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        """
        node_posterior = y_neighbor_distr
        return node_posterior

    def get_virual_link_proba(self, node_posterior: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes virtual link probabilities based on node posterior likelihood.

        Parameters:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - virtual_link_proba: torch.Tensor
            Virtual link probabilities.
        """

        node_posterior *= 1 - F.one_hot(y_pred, num_classes=self.n_class)

        node_posterior = node_posterior.clip(min=0)

        node_posterior /= node_posterior.sum(axis=1).reshape(-1, 1)

        virtual_link_proba = node_posterior.nan_to_num(0)
        return virtual_link_proba