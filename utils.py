import torch
from einops import rearrange


def create_mask(window_size, displacement, up_to_down, left_to_right):
    """
    Create the attention mask which will be added to the scaled dot product attentions of the last
    layer and last column. This attention mask prevents the calculation of attention between spatially
    unrelated tokens within a patch.
    Input:
        - window_size: the size of each window patch; int
        - displacement: how much displacement was used in the cyclic shift; int
        - up_to_down: if the mask is meant for the last row; bool
        - left_to_right: if the mask is meant for the last column; bool
    Output:
        - mask: matrix of shape (WINDOW_SIZE^2, WINDOW_SIZE^2) with 0 or -inf as possible values. The
        positions which should be masked correspond to a value of -inf in the attention mask matrix.
    """
    mask = torch.zeros(window_size ** 2, window_size ** 2)  # (WINDOW_SIZE^2, WINDOW_SIZE^2)
    if up_to_down:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_to_right:
        mask = rearrange(mask, '(h1 h2) (w1 w2) -> h1 h2 w1 w2', h1=window_size, w1=window_size)
        # # (WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 h2 w1 w2 -> (h1 h2) (w1 w2)')  # # (WINDOW_SIZE^2, WINDOW_SIZE^2)
    return mask


def get_relative_distances(window_size):
    """
    Create a (WINDOW_SIZE^2, WINDOW_SIZE^2, 2) matrix which holds for each element the relative
    distance to every other element in a patch of size (WINDOW_SIZE, WINDOW_SIZE).
    Input:
        - window_size: the size of the window patch (either height or width); int
    Output:
        - distances: matrix of shape (WINDOW_SIZE^2, WINDOW_SIZE^2, 2)
    Example:
        - input: 2
        - output: matrix "distances" of shape (4, 4, 2). Element distances[1] would be
        [[0, -1], [0, 0], [1, -1], [1, 0]] which correspond to the 2D representation:

        [[0, -1]  [0, 0]
         [1, -1]  [1, 0]]
         which makes sense since the origin in this case is the upper right corner and the matrix
         is populated with the relative distances from this origin to every other position
    """
    x = torch.arange(window_size)  # (WINDOW_SIZE)
    y = torch.arange(window_size)  # (WINDOW_SIZE)
    indices = torch.stack(torch.meshgrid([x, y], indexing="ij"))  # (2, WINDOW_SIZE, WINDOW_SIZE)
    indices = torch.flatten(indices, start_dim=1)  # (2, WINDOW_SIZE^2)
    indices = indices.transpose(0, 1)  # (WINDOW_SIZE^2, 2)
    distances = indices[None, :, :] - indices[:, None, :]  # (WINDOW_SIZE^2, WINDOW_SIZE^2, 2)
    return distances
