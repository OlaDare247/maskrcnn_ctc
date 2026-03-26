import numpy as np
import cv2
import torch
import torch.nn.functional as F

def farneback_flow(prev_u8: np.ndarray, curr_u8: np.ndarray) -> np.ndarray:
    """
    prev_u8, curr_u8: HxW uint8
    returns flow: HxWx2 float32, flow[y,x] = (dx, dy)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_u8, curr_u8, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow.astype(np.float32)

def warp_logits_with_flow(logits: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
    """
    logits: [1, C, H, W] torch float (CUDA or CPU)
    flow:   HxWx2 numpy float32 (dx,dy) mapping prev->curr
    returns warped logits: [1,C,H,W]
    """
    device = logits.device
    _, _, H, W = logits.shape

    # Build normalized sampling grid for grid_sample
    # grid_sample wants grid in normalized coords [-1,1]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xx2 = xx + flow[..., 0]
    yy2 = yy + flow[..., 1]

    # normalize
    grid_x = 2.0 * (xx2 / (W - 1)) - 1.0
    grid_y = 2.0 * (yy2 / (H - 1)) - 1.0
    grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # HxWx2

    grid_t = torch.from_numpy(grid).to(device=device)
    grid_t = grid_t.unsqueeze(0)  # 1xHxWx2

    warped = F.grid_sample(
        logits, grid_t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    return warped
