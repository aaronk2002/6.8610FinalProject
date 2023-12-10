import torch
import argparse


def evaluate_probes(num_layers, task_type, loss_fn):
    """
    Given the model and dataset paths, and the loss function, evaluate
    the training and evaluation losses for each models at each layer

    Input:
    - num_layers    : number of layers in transformer models
    - task_type     : type of task to evaluate
    - loss_fn       : the loss function to evaluate how well the model
                      performs

    Return:
    Two lists of losses for each train and eval sets, each loss corresponding
    to a probe model in the model_path
    """
    # Get model and dataset
    task_to_idx = {"composer": 0, "key": 1, "control": 2}
    models = torch.load(f"{task_type}.pth", map_location=torch.device("cpu"))
    dataset = torch.load(
        f"../dataset/{num_layers}-layers-probe.pth", map_location=torch.device("cpu")
    )
    train_x, train_y, eval_x, eval_y = (
        dataset["train_x"],
        dataset["train_y"][:, task_to_idx[task_type]],
        dataset["eval_x"],
        dataset["eval_y"][:, task_to_idx[task_type]],
    )
    if task_type != "control":
        train_y = train_y.to(torch.long)
        eval_y = eval_y.to(torch.long)
    model = models[num_layers]

    # Prediction
    train_y_pred = []
    eval_y_pred = []
    for idx in range(num_layers):
        train_y_pred.append(model[idx](train_x[idx]))
        eval_y_pred.append(model[idx](eval_x[idx]))

    # Evaluation
    train_y_loss = []
    eval_y_loss = []
    for idx in range(num_layers):
        train_y_loss.append(loss_fn(train_y_pred[idx], train_y))
        eval_y_loss.append(loss_fn(eval_y_pred[idx], eval_y))

    return train_y_loss, eval_y_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", help="number of layers", type=int)
    parser.add_argument("--task", help="type of task: control, key, composer", type=str)
    args = parser.parse_args()

    def regression_loss(y_1, y_2):
        mse = torch.nn.MSELoss()(y_1.view(-1), y_2).item()
        y_2_mean = torch.mean(y_2)
        r_squared = 1 - mse / torch.mean((y_2 - y_2_mean) * (y_2 - y_2_mean))
        return {"Mean Squared Error": mse, "R^2": r_squared}

    def classification_loss(y_1, y_2):
        cell = torch.nn.CrossEntropyLoss()(y_1, y_2)
        acc = sum(torch.argmax(y_1, dim=1) == y_2) / len(y_2)
        return {"Cross Entropy": cell.item(), "Accuracy": acc.item()}

    print(
        evaluate_probes(
            args.layers,
            args.task,
            regression_loss if args.task == "control" else classification_loss,
        )
    )
