from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg
from typing import List, Tuple


from torchvision.transforms import Compose, Normalize, ToTensor,Resize

from cnn_img.task import Net, get_weights, set_weights, test
from cnn_img.strategy import CustomFedAvg
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 5:
        lr /= 2
    return {"lr": lr}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

<<<<<<< HEAD
    global_test_data = load_dataset("sarath2003/BreakHis")
=======
    global_test_data = load_dataset("uoft-cs/cifar10")["test"]
>>>>>>> fbf5f0b (update included remove folder wandb and outputs)
    transfrom = Compose(
        [Resize((400, 700)),ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [transfrom(img) for img in batch["image"]]
        return batch
    
    train_test_split = global_test_data["train"].train_test_split(test_size=0.2, seed=42)

    testloader = DataLoader(
        train_test_split["test"].with_transform(apply_transforms), batch_size=32
    )

    # Initialize model parameters
    ndarrays = get_weights(
        Net(),
    )
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=0.25,
        min_available_clients=5,
        initial_parameters=parameters,
        # proximal_mu=0.3,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(
            testloader, device=context.run_config["server-device"]
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
<<<<<<< HEAD
        # eta = 0.015,
        # eta_l = 0.03,
=======
        eta=0.15,
        eta_l=0.01,
>>>>>>> fbf5f0b (update included remove folder wandb and outputs)
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
