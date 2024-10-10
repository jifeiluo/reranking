import torch
import feature_operations

def neighbor_guided_similarity_smooth(confident_matrix, candidate_matrix, semi_candidate_matrix, trans_matrix, beta=0.0):
    
    confident_non_zero_all_index = torch.nonzero(confident_matrix, as_tuple=True)[1]

    confident_non_zero_num = torch.count_nonzero(confident_matrix, dim=1)
    candidate_non_zero_num = torch.count_nonzero(candidate_matrix, dim=1)
    semi_candidate_non_zero_num = torch.count_nonzero(semi_candidate_matrix, dim=1)
    confident_non_zero_start_num = torch.cumsum(confident_non_zero_num) - confident_non_zero_num
    confident_non_zero_end_num = confident_non_zero_start_num + confident_non_zero_num - 1
    semi_candidate_non_zero_start_num = torch.cumsum(semi_candidate_non_zero_num) - semi_candidate_non_zero_num
    semi_candidate_non_zero_end_num = semi_candidate_non_zero_start_num + semi_candidate_non_zero_num

    anchor_vector = feature_operations.compute_anchor_vector(trans_matrix, 
                                                             confident_non_zero_start_num, 
                                                             confident_non_zero_end_num, 
                                                             confident_non_zero_all_index)
    transition_weight_matrix = feature_operations.compute_transition_weight_matrix_(trans_matrix, 
                                                                                    confident_non_zero_start_num,
                                                                                    confident_non_zero_end_num,
                                                                                    confident_non_zero_all_index,
                                                                                    semi_candidate_non_zero_start_num,
                                                                                    semi_candidate_non_zero_end_num,
                                                                                    candidate_matrix,
                                                                                    anchor_vector)

    tmp_value = (torch.pow(anchor_vector, 2)*torch.sum(sparse_feature, dim=1) - anchor_vector*torch.sum(torch.mul(sparse_feature, transition_weight_matrix), dim=1)) / candidate_non_zero_num
    sparse_feature = (sparse_feature*(torch.mul(anchor_vector.unsqueeze(1), transition_weight_matrix)+2*beta) + tmp_value.unsqueeze(1)) / (torch.pow(anchor_vector, 2).unsqueeze(1)+2*beta)
