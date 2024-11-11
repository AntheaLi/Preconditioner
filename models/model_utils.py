import torch


def lift(scores_source, scores_target, nodes_features_matrix_proj, edge_index, nodes_dim=0):
    """
    Lifts i.e. duplicates certain vectors depending on the edge index.
    One of the tensor dims goes from N -> E (that's where the "lift" comes from).

    """
    src_nodes_index = edge_index[nodes_dim]
    trg_nodes_index = edge_index[nodes_dim]

    # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
    scores_source = scores_source.index_select(nodes_dim, src_nodes_index)
    scores_target = scores_target.index_select(nodes_dim, trg_nodes_index)
    nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(nodes_dim, src_nodes_index)

    return scores_source, scores_target, nodes_features_matrix_proj_lifted

