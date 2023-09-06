import os

import numpy as np
import torch
from tqdm import trange

from core.nn.mlp_tmpred import MLPTMPred
from core.traffic_routing.te_util import load_network_topology
from core.utils.data_utils.data_loading import load_data_tm_pred


def training_tm_pred(args):
    data_loader = load_data_tm_pred(args)
    use_orthogonal = args.use_orthogonal
    use_ReLU = args.use_ReLU

    graph = load_network_topology(args.dataset, args.data_folder)

    num_node = graph.number_of_nodes()
    num_link = graph.number_of_edges()
    model = MLPTMPred(input_dim=[num_node * num_node, num_link], output_dim=num_node * num_node, hidden_size=64,
                      layer_N=1, use_link_u=True, use_orthogonal=use_orthogonal, use_ReLU=use_ReLU)

    model.to(args.device)
    criterion = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

    num_epoch = 200
    epoch_losses = []
    min_val_loss = np.inf
    num_not_improved = 0
    iterations = trange(num_epoch)
    for i in iterations:
        epoch_loss = training(model, data_loader['train'], optimizer, criterion)
        epoch_losses.append(epoch_loss)
        val_loss = validating(model, data_loader['val'], criterion)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            num_not_improved = 0
            save_model(model, args)
        else:
            num_not_improved += 1

        if num_not_improved > int(num_epoch / 2):
            break

        iterations.set_description(f'Epoch {i} loss {epoch_loss} val_loss {val_loss} patience {num_not_improved}')
    return model


def training(model, loader, optimizer, criterion):
    model.train()
    batch_loss = []

    for idx, (x_tm, x_lu, y) in enumerate(loader):
        # zero grad
        model.zero_grad()
        # inference
        y_hat = model(x_tm, x_lu)
        if y_hat.size() != y.size():
            y_hat = torch.reshape(y_hat, y.size())

        # update model
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        # info
        batch_loss.append(loss.item())
    # info
    return np.sum(batch_loss) / len(batch_loss)


def validating(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        batch_loss = []
        for idx, (x_tm, x_lu, y) in enumerate(loader):
            # zero grad
            # inference
            y_hat = model(x_tm, x_lu)
            if y_hat.size() != y.size():
                y_hat = torch.reshape(y_hat, y.size())

            # update model
            loss = criterion(y_hat, y)
            # info
            batch_loss.append(loss.item())
        # info
        return np.sum(batch_loss) / len(batch_loss)


def save_model(model, args):
    save_path = os.path.join(args.data_folder, 'tm_pred_model')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{args.dataset}_best_model.pkl')
    torch.save(model.state_dict(), save_file)


def load_tm_pred_model(args):
    save_path = os.path.join(args.data_folder, 'tm_pred_model')
    save_file = os.path.join(save_path, f'{args.dataset}_best_model.pkl')
    use_orthogonal = args.use_orthogonal
    use_ReLU = args.use_ReLU

    graph = load_network_topology(args.dataset, args.data_folder)

    num_node = graph.number_of_nodes()
    num_link = graph.number_of_edges()

    model = MLPTMPred(input_dim=[num_node * num_node, num_link], output_dim=num_node * num_node, hidden_size=64,
                      layer_N=1, use_link_u=True, use_orthogonal=use_orthogonal, use_ReLU=use_ReLU)
    model_state = torch.load(save_file)
    model.load_state_dict(model_state)
    model.to(args.device)
    return model
