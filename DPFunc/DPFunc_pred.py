import os
import dgl
import torch
import warnings
import numpy as np
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
from DPFunc.models import combine_inter_model
from DPFunc.model_utils import test_performance_gnn_inter, merge_result
from typing import List, Sequence, Optional, Dict, Any, Union


def dpfunc_predict_in_memory(
    *,
    ont: str,
    pid_list: Sequence[str],
    graphs: Sequence[dgl.DGLGraph],
    interpro,  # CSR preferred; dense allowed but will be converted to CSR
    mlb,
    checkpoint_paths: Sequence[str],
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 64,
    head: int = 4,
    inter_hid: int = 1280,
    graph_size: int = 1280,
    graph_hid: int = 1280,
    save_each_submodel: bool = False,
    save_files: Optional[Sequence[str]] = None,
):

    if ont not in {"bp", "mf", "cc"}:
        raise ValueError("ont must be one of: bp, mf, cc")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    pid_list = list(pid_list)
    graphs = list(graphs)

    if len(pid_list) != len(graphs):
        raise ValueError(f"len(pid_list)={len(pid_list)} must equal len(graphs)={len(graphs)}")

    # --- InterPro MUST be CSR for DPFunc's test_performance_gnn_inter (.indices/.data) ---
    if sp.isspmatrix(interpro):
        interpro_csr = interpro.tocsr().astype(np.float32)
    elif isinstance(interpro, torch.Tensor):
        interpro_csr = sp.csr_matrix(interpro.detach().cpu().numpy().astype(np.float32))
    else:
        interpro_csr = sp.csr_matrix(np.asarray(interpro, dtype=np.float32))

    if interpro_csr.ndim != 2 or interpro_csr.shape[0] != len(pid_list):
        raise ValueError(
            f"interpro must be shape [N, inter_size]; got {interpro_csr.shape}, N={len(pid_list)}"
        )

    dummy_go_by_ont = {
        "mf": ["GO:0003674"],
        "bp": ["GO:0008150"],
        "cc": ["GO:0005575"],
    }
    test_go = [dummy_go_by_ont[ont] for _ in pid_list]

    labels_num = len(mlb.classes_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_y = mlb.transform(test_go).astype(np.float32)

    idx_goid = {i: go for i, go in enumerate(mlb.classes_)}
    goid_idx = {go: i for i, go in enumerate(mlb.classes_)}

    test_data = [(graphs[i], i, test_y[i]) for i in range(len(test_y))]
    test_dataloader = GraphDataLoader(
        test_data, batch_size=batch_size, drop_last=False, shuffle=False
    )

    model = combine_inter_model(
        inter_size=interpro_csr.shape[1],
        inter_hid=inter_hid,
        graph_size=graph_size,
        graph_hid=graph_hid,
        label_num=labels_num,
        head=head,
    ).to(device)

    cob_pred_df = []

    if save_each_submodel and (not save_files or len(save_files) != len(checkpoint_paths)):
        raise ValueError("If save_each_submodel=True, provide save_files with same length as checkpoint_paths")

    for k, ckpt_path in enumerate(checkpoint_paths):
        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        pred_df = test_performance_gnn_inter(
            model,
            test_dataloader,
            pid_list,
            interpro_csr,   # <- keep sparse
            test_y,
            idx_goid,
            goid_idx,
            ont,
            device,
            save=save_each_submodel,
            save_file=(save_files[k] if save_each_submodel else None),
            evaluate=False,
        )
        cob_pred_df.append(pred_df)

    if not cob_pred_df:
        raise FileNotFoundError("No checkpoints were found/loaded from checkpoint_paths")

    return merge_result(cob_pred_df)