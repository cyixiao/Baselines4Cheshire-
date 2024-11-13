from utils import *
import torch
from CHESHIRE import CHESHIRE
from NHP import NHP
from HGNN import HGNN
from tqdm import tqdm
import config
import pandas as pd
import cobra
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

args = config.parse()

def train(feature, y, incidence_matrix, model, optimizer):
    model.train()
    optimizer.zero_grad()
    y_pred = model(feature, incidence_matrix)
    loss = hyperlink_score_loss(y_pred, y)
    print(loss.cpu())
    loss.backward()
    optimizer.step()


def predict(feature, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = model(feature, incidence_matrix)
    return torch.squeeze(y_pred)

# def load_gem_data(file_path):
#     model = cobra.io.read_sbml_model(file_path)
#     reactions = model.reactions
#     rxn_ids = [rxn.id for rxn in reactions]
#     rxn_matrix = cobra.util.create_stoichiometric_matrix(model, array_type='DataFrame')
#     rxn_df = rxn_matrix[rxn_ids]
#     return rxn_df

def load_gem_data(file_path):
    model = cobra.io.read_sbml_model(file_path)
    reactions = model.reactions
    metabolites = model.metabolites
    rxn_ids = [rxn.id for rxn in reactions]
    met_ids = [met.id for met in metabolites]
    rxn_matrix = cobra.util.create_stoichiometric_matrix(model, array_type='DataFrame')
    rxn_matrix.columns = rxn_ids
    rxn_matrix.index = met_ids
    
    return rxn_matrix


def create_label(incidence_matrix_pos, incidence_matrix_neg):
    pos_labels = torch.ones(incidence_matrix_pos.size(1), dtype=torch.float)
    neg_labels = torch.zeros(incidence_matrix_neg.size(1), dtype=torch.float)
    labels = torch.cat((pos_labels, neg_labels), dim=0)
    return labels

def get_prediction_score(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = f'./data/{name}/{name}.xml'
    rxn_df = load_gem_data(file_path)
    pos_matrix = np.abs(rxn_df.to_numpy()) > 0
    pos_tensor = torch.tensor(pos_matrix, dtype=torch.float)

    # pos_df = pd.DataFrame(pos_tensor.numpy(), index=rxn_df.index, columns=rxn_df.columns)
    pos_df = pd.DataFrame(pos_tensor.numpy())
    
    train_split = args.train_split if args.recover else 0.6
    valid_split = train_split + 0.1 if args.recover else 0.8

    if args.recover:
        train_pos_df, test_pos_df = np.split(
            pos_df.sample(frac=1, axis=1, random_state=args.seed), 
            [int(train_split * len(pos_df.columns))],
            axis=1
        )
        
        train_pos = torch.tensor(train_pos_df.to_numpy(), dtype=torch.float).to(device)
        test_pos = torch.tensor(test_pos_df.to_numpy(), dtype=torch.float).to(device)

        train_neg = create_neg_incidence_matrix(train_pos).to(device)

        train_data = torch.cat((train_pos, train_neg), dim=1).to(device)
        y_train = create_label(train_pos, train_neg).to(device)

        test_data = test_pos
        y_test = torch.ones(test_pos.shape[1], dtype=torch.float).to(device)

    else:
        train_pos_df, valid_pos_df, test_pos_df = np.split(
            pos_df.sample(frac=1, axis=1, random_state=args.seed), 
            [int(train_split * len(pos_df.columns)), int(valid_split * len(pos_df.columns))],
            axis=1
        )
        
        train_pos = torch.tensor(train_pos_df.to_numpy(), dtype=torch.float).to(device)
        test_pos = torch.tensor(test_pos_df.to_numpy(), dtype=torch.float).to(device)

        train_neg = create_neg_incidence_matrix(train_pos).to(device)
        test_neg = create_neg_incidence_matrix(test_pos).to(device)

        train_data = torch.cat((train_pos, train_neg), dim=1).to(device)
        test_data = torch.cat((test_pos, test_neg), dim=1).to(device)

        y_train = create_label(train_pos, train_neg).to(device)
        y_test = create_label(test_pos, test_neg).to(device)


    # train_pos_df, valid_pos_df, test_pos_df = np.split(
    #     pos_df.sample(frac=1, axis=1, random_state=args.seed), 
    #     [int(train_split * len(pos_df.columns)), int(valid_split * len(pos_df.columns))],
    #     axis=1
    # )

    # if args.recover:
    #     universe_df = load_gem_data('./data/pool/bigg_universe.xml')
    #     full_metabolite_set = universe_df.index.union(train_pos_df.index)
        
    #     train_pos_df = train_pos_df.reindex(full_metabolite_set, fill_value=0)
    #     test_pos_df = test_pos_df.reindex(full_metabolite_set, fill_value=0)
        
    #     train_neg = create_neg_incidence_matrix(torch.tensor(train_pos_df.values, dtype=torch.float)).to(device)
    #     universe_neg_df = universe_df.reindex(full_metabolite_set, fill_value=0).drop(columns=test_pos_df.columns, errors='ignore')
    #     test_neg_df = universe_neg_df.sample(n=test_pos_df.shape[1], axis=1, random_state=args.seed)
    #     test_neg = torch.tensor(np.abs(test_neg_df.to_numpy()) > 0, dtype=torch.float).to(device)

    # else:
    #     train_pos = torch.tensor(train_pos_df.to_numpy(), dtype=torch.float).to(device)
    #     test_pos = torch.tensor(test_pos_df.to_numpy(), dtype=torch.float).to(device)

    #     train_neg = create_neg_incidence_matrix(train_pos).to(device)
    #     test_neg = create_neg_incidence_matrix(test_pos).to(device)

    # train_pos = torch.tensor(train_pos_df.to_numpy(), dtype=torch.float).to(device)
    # test_pos = torch.tensor(test_pos_df.to_numpy(), dtype=torch.float).to(device)

    # train_data = torch.cat((train_pos, train_neg), dim=1).to(device)
    # test_data = torch.cat((test_pos, test_neg), dim=1).to(device)
    
    # y_train = create_label(train_pos, train_neg).to(device)
    # y_test = create_label(test_pos, test_neg).to(device)

    if args.model == 'CHESHIRE':
        model = CHESHIRE(input_dim=train_pos.shape, emb_dim=args.emb_dim, conv_dim=args.conv_dim, k=args.k, p=args.p).to(device)
    elif args.model == 'NHP':
        model = NHP(input_dim=train_pos.shape, emb_dim=args.emb_dim, conv_dim=args.conv_dim).to(device)
    elif args.model == 'HGNN':
        model = HGNN(input_dim=train_pos.shape, hidden_dim=args.emb_dim, n_class=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in tqdm(range(args.max_epoch)):
                train(train_pos, y_train, train_data, model, optimizer)

    y_pred_test = predict(train_pos, test_data, model)

    y_pred_test = y_pred_test.cpu().numpy()
    test_labels = y_test.cpu().numpy()
    try:
        auc = roc_auc_score(test_labels, y_pred_test)
    except ValueError:
        auc = None
    f1 = f1_score(test_labels, np.round(y_pred_test))
    accuracy = accuracy_score(test_labels, np.round(y_pred_test))

    # top_n = int(y_test.sum().item())
    # top_n_indices = np.argsort(y_pred_test)[-top_n:]
    # top_n_true_labels = test_labels[top_n_indices]
    # top_n_pred_labels = np.round(y_pred_test[top_n_indices])
    # recover_rate = accuracy_score(top_n_true_labels, top_n_pred_labels)

    b_score = np.round(y_pred_test)
    TP = (b_score == 1).sum()
    recover_rate = TP / len(test_labels)

    auc_display = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"Model: {name}, AUC: {auc_display}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Recover Rate: {recover_rate:.4f}")

    return auc, f1, accuracy, recover_rate
