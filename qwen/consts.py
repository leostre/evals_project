# ====================== CONFIG ======================
MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE = "cuda"
BATCH_SIZE = 1          # Very conservative for 8B
EPOCHS = 16
LR = 5e-4

LAYERS = list(range(0, 36, 2))

ACTIVATION_TYPES = [
    "resid_post",
    'resid_mid',
                     "mlp_post", 
                     "attn_out"
                     ]  # Add more as needed



AGG_FUNCS_ACTIVATIONS = {
    'mean': lambda x: x.mean(dim=1).cpu(), 
    'last': lambda x: x[:, -1, :].cpu()
}
