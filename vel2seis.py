import torch
import numpy as np

def ricker_t(f, dt, nt):
    """
    Generate a differentiable Ricker wavelet using PyTorch.
    Supports autograd for f and dt.

    Args:
        f (Tensor): Dominant frequency (can require grad).
        dt (Tensor): Time step (can require grad).
        nt (int): Number of time steps.

    Returns:
        Tuple[Tensor, Tensor]:
            - w: Wavelet (nt,)
            - tw: Time axis (nt,)
    """
    # Approximate the number of wavelet samples based on frequency and time step
    approx_nw = 2.2 / (f * dt)
    nw_float = 2.0 * torch.floor(approx_nw / 2.0) + 1  # Ensure odd size
    nw = nw_float.to(dtype=torch.int32).item()  # Convert to integer for indexing

    # Center index of wavelet
    nc = nw // 2
    k = torch.arange(nw, dtype=torch.float32, device=f.device if isinstance(f, torch.Tensor) else None)

    # Compute wavelet samples
    alpha = (nc - k) * f * dt * torch.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * torch.exp(-beta)

    if nt < nw:
        raise ValueError("nt must be >= nw")

    # Zero-pad the wavelet to full length
    w = torch.zeros(nt, dtype=w0.dtype, device=w0.device)
    w[:nw] = w0
    w = w.clone()  # Clone to preserve autograd behavior

    # Time axis
    tw = torch.arange(nt, dtype=torch.float32, device=w.device) * dt
    return w, tw

def padvel_t(v0, nbc):
    """
    Apply edge-replication padding to a 2D tensor.

    Args:
        v0 (Tensor): Input velocity model (H, W)
        nbc (int): Number of boundary cells to pad.

    Returns:
        Tensor: Padded velocity model (H+2*nbc, W+2*nbc)
    """
    pad = torch.nn.ReplicationPad2d(nbc)
    v0 = v0.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    out = pad(v0)
    return out.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def expand_source_t(s0, nt):
    """
    Zero-pad a 1D source wavelet to a fixed time length.

    Args:
        s0 (array-like): Input wavelet.
        nt (int): Target number of time steps.

    Returns:
        Tensor: Zero-padded wavelet (nt,)
    """
    s0 = torch.as_tensor(s0, dtype=torch.float32).flatten()
    ns = s0.shape[0]

    if ns > nt:
        raise ValueError("s0 is longer than nt")

    s = torch.zeros(nt, dtype=s0.dtype, device=s0.device)
    s[:ns] = s0
    return s.clone()

def AbcCoef2D_t(vel, nbc, dx):
    """
    Generate absorbing boundary damping coefficients for a 2D velocity model.

    Args:
        vel (Tensor): Velocity model with padding (nzbc, nxbc)
        nbc (int): Number of boundary cells
        dx (float): Spatial resolution

    Returns:
        Tensor: Damping coefficients (nzbc, nxbc)
    """
    nzbc, nxbc = vel.shape
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    # Minimum velocity used to scale damping
    velmin = torch.min(vel)
    a = (nbc - 1) * dx  # Effective thickness of absorbing layer
    kappa = 3.0 * velmin * torch.log(torch.tensor(1e7, dtype=vel.dtype, device=vel.device)) / (2.0 * a)

    # Damping profile from edge to interior
    idx = torch.arange(nbc, dtype=vel.dtype, device=vel.device)
    damp1d = kappa * ((idx * dx / a) ** 2)

    # Initialize damping matrix
    damp = torch.zeros((nzbc, nxbc), dtype=vel.dtype, device=vel.device)

    # Left and right edges
    damp[:, :nbc] = damp1d.flip(0).unsqueeze(0)
    damp[:, nx + nbc:] = damp1d.unsqueeze(0)

    # Top and bottom edges
    damp[:nbc, nbc:nx + nbc] = damp1d.flip(0).unsqueeze(1)
    damp[nz + nbc:, nbc:nx + nbc] = damp1d.unsqueeze(1)

    return damp

def corrected_reception(p, igx, igz):
    """
    Extract receiver data from pressure field using integer grid indices.

    Args:
        p (Tensor): Pressure field (B, H, W)
        igx, igz (Tensor): Receiver grid indices (B, N)

    Returns:
        Tensor: Receiver data (B, N)
    """
    B, H, W = p.shape
    nx = igx.shape[1]
    batch_idx = torch.arange(B, device=p.device).view(B, 1).expand(B, nx)
    return p[batch_idx, igz.long(), igx.long()]

def prepare_geom_multi_source(source_positions, dx, nbc, nx, device):
    """
    Prepare grid indices for multiple sources and receivers.

    Args:
        source_positions (list): Source x positions in index units
        dx (float): Grid spacing
        nbc (int): Number of absorbing cells
        nx (int): Number of receiver x positions
        device (str): Device to use

    Returns:
        Dict[str, Tensor]: Contains source and receiver positions and indices.
    """
    dtype = torch.float32
    ns = len(source_positions)

    sx = torch.tensor(source_positions, dtype=dtype, device=device) * dx
    sz = torch.full((ns,), dx, dtype=dtype, device=device)
    isx = sx / dx + nbc
    isz = sz / dx + nbc
    isx_i = torch.round(isx).long()
    isz_i = torch.round(isz).long()

    gx = torch.arange(nx, dtype=dtype, device=device) * dx
    gz = torch.full((nx,), dx, dtype=dtype, device=device)
    igx = torch.round(gx / dx + nbc).long()
    igz = torch.round(gz / dx + nbc).long()

    igx_all = igx.unsqueeze(0).expand(ns, -1)
    igz_all = igz.unsqueeze(0).expand(ns, -1)

    return {
        "isx": isx, "isz": isz,
        "isx_i": isx_i, "isz_i": isz_i,
        "igx": igx_all, "igz": igz_all
    }

def a2d_mod_abc24_t_batched(vel, nbc, dx, nt, dt, s, isx_i, isz_i, igx, igz, isFS=False):
    """
    Batched version of 2D wave propagation using finite difference and ABC.

    Args:
        vel (Tensor): Velocity model (nz, nx)
        nbc (int): Padding size
        dx, dt (float): Space and time steps
        nt (int): Number of time steps
        s (Tensor): Source wavelet (nt,)
        isx_i, isz_i (Tensor): Source indices (B,)
        igx, igz (Tensor): Receiver indices (B, nx)

    Returns:
        Tensor: Seismograms (B, nt, nx)
    """
    B = isx_i.shape[0]
    nx = igx.shape[1]
    device = vel.device
    dtype = vel.dtype

    # Padding and damping
    v_pad = padvel_t(vel, nbc)
    abc = AbcCoef2D_t(v_pad, nbc, dx)

    # Finite difference coefficients
    alpha = (v_pad * dt / dx) ** 2
    kappa = abc * dt
    temp1 = 2 + 2 * (-2.5) * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v_pad * dt) ** 2

    # Expand for batch
    v_pad = v_pad.expand(B, -1, -1)
    alpha = alpha.expand(B, -1, -1)
    temp1 = temp1.expand(B, -1, -1)
    temp2 = temp2.expand(B, -1, -1)
    beta_dt = beta_dt.expand(B, -1, -1)

    p0 = torch.zeros_like(v_pad)
    p1 = torch.zeros_like(v_pad)
    seis = torch.zeros((B, nt, nx), dtype=dtype, device=device)
    s = expand_source_t(s, nt)

    batch_indices = torch.arange(B, device=device)

    # Main time loop
    for it in range(nt):
        c2, c3 = 4.0 / 3.0, -1.0 / 12.0
        p = (temp1 * p1 - temp2 * p0 +
             alpha * (
                 c2 * (torch.roll(p1, 1, dims=2) + torch.roll(p1, -1, dims=2) +
                        torch.roll(p1, 1, dims=1) + torch.roll(p1, -1, dims=1)) +
                 c3 * (torch.roll(p1, 2, dims=2) + torch.roll(p1, -2, dims=2) +
                        torch.roll(p1, 2, dims=1) + torch.roll(p1, -2, dims=1))
             ))

        # Inject source
        p[batch_indices, isz_i, isx_i] += beta_dt[batch_indices, isz_i, isx_i] * s[it]
        # Record receivers
        seis[:, it] = p[batch_indices.view(-1, 1), igz, igx]
        p0, p1 = p1, p

    seis = seis.astype(torch.float16)
    return seis

class SeismicGeometry:
    """
    Object-oriented wrapper for 2D seismic simulation with multiple sources.

    Improvements over the basic version:
    - Differentiable PyTorch implementation (autograd-compatible)
    - Batched wave propagation for multiple sources
    - Object encapsulation for reuse and clarity
    """
    def __init__(
        self,
        source_positions=[0, 17, 34, 52, 69],
        nbc=120,
        nx=70,
        dx=10.0,
        freq=15.0,
        dt=1e-3,
        nt=1001,
        isFS=False,
        device="cuda",
        dtype=torch.float32
    ):
        self.device = device
        self.dtype = dtype
        self.source_positions = source_positions
        self.geom = prepare_geom_multi_source(
            source_positions, dx, nbc, nx, device
        )
        self.nbc = nbc
        self.dx = torch.tensor(dx, dtype=dtype, device=device)
        self.dt = torch.tensor(dt, dtype=dtype, device=device)
        self.freq = torch.tensor(freq, dtype=dtype, device=device)
        self.nt = int(nt)
        self.s, _ = ricker_t(self.freq, self.dt, self.nt)
        self.isFS = False

    def simulate(self, vel):
        """
        Run the seismic simulation.

        Args:
            vel (Tensor): Velocity model (70, 70)

        Returns:
            Tensor: Seismograms (5, 1001, 70)
        """
        return a2d_mod_abc24_t_batched(
            vel, self.nbc, self.dx, self.nt, self.dt, self.s,
            self.geom["isx_i"], self.geom["isz_i"],
            self.geom["igx"], self.geom["igz"],
            self.isFS
        )