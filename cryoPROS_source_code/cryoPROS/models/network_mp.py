import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import mrcfile

def compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    """
    Compute the 2D CTF
    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """
        
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2) ** 0.5
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.atan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
        2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
        - phase_shift
    )
    ctf = (1 - w**2) ** 0.5 * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    
    return ctf


def discrete_radon_transform_3d(volume, rotation):
    b = volume.shape[0]
    
    zeros = torch.zeros(b, 3, 1).to(volume.device)
    theta = torch.cat([rotation, zeros], dim=2)

    grid = F.affine_grid(theta, size=volume.shape)
    volume_rot = F.grid_sample(volume, grid, mode='bilinear')
    
    volume_rot = volume_rot.permute(0, 1, 3, 4, 2)
    proj = volume_rot.sum(dim=-1)
    
    return proj


def translation_2d(proj, trans):
    """
    Input:
        proj: Bx1xbsxbs tensor 
        trans: Bx2 tensor
    """
    
    b = trans.shape[0]
    
    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device)
    trans = trans.unsqueeze(-1)
    theta = torch.cat([eye, trans], dim=2)

    grid = F.affine_grid(theta, size=proj.shape)
    proj_trans = F.grid_sample(proj, grid, mode='bicubic')
    
    return proj_trans


def convolve_ctf(proj, ctf):
    Fproj = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    Fproj = Fproj * ctf
    Fproj = torch.fft.ifftshift(Fproj, dim=(-2, -1))
    proj = torch.fft.ifft2(Fproj)
    proj = proj.real
    
    return proj


def init_volume(box_size):
    volume = torch.zeros(box_size, box_size, box_size)
    volume[box_size // 4 : 3 * box_size // 4, 
           box_size // 4 : 3 * box_size // 4,
           box_size // 4 : 3 * box_size // 4] = torch.ones(box_size//2, box_size//2, box_size//2)
    
    volume = volume / torch.norm(volume)
    
    return volume


class Reconstructor(nn.Module):
    def __init__(self, box_size=256, Apix=1.31, invert=True, init_volume_path=None,
                 volume_scale=1, update_volume_scale=False, update_volume=False, mask_path=None):
        super(Reconstructor, self).__init__()
        
        if init_volume_path is not None:
            with mrcfile.open(init_volume_path, permissive=True) as mrc:
                volume = mrc.data
            volume = torch.from_numpy(volume.astype(np.float32))
        else:
            volume = init_volume(box_size)
        
        volume = volume / torch.norm(volume)
        if update_volume:
            self.volume = nn.Parameter(volume, requires_grad=True)
        else:
            self.volume = nn.Parameter(volume, requires_grad=False)
        
        self.volume_scale = nn.Parameter(torch.Tensor([volume_scale]), requires_grad=bool(update_volume_scale))
        
        micelle = init_volume(box_size)
        self.micelle = nn.Parameter(micelle, requires_grad=True)
        
        if mask_path is not None:
            with mrcfile.open(mask_path, permissive=True) as mrc:
                mask = mrc.data
        else:
            mask = np.ones((box_size, box_size, box_size))
        mask = torch.from_numpy(mask).float()
        self.mask = nn.Parameter(mask, requires_grad=False)                
                
        freqs = (
        np.stack(
            np.meshgrid(
                np.linspace(-0.5, 0.5, box_size, endpoint=False),
                np.linspace(-0.5, 0.5, box_size, endpoint=False),
            ),
            -1,
        )
        / Apix
        )
        freqs = freqs.reshape(-1, 2)
        self.freqs = torch.from_numpy(freqs).unsqueeze(0).float()
        self.box_size = box_size
        self.invert = invert


    def forward(self, img, rotation, trans, ctf_para):
    
        b = rotation.shape[0]
        volume = self.get_volume()
        volume = volume.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        proj = discrete_radon_transform_3d(volume, rotation)
        proj = translation_2d(proj, 2 * trans)
        
        freqs = self.freqs.to(img.device).repeat(b, 1, 1)
        ctf = compute_ctf(freqs, *torch.split(ctf_para, 1, 1))
        ctf = ctf.reshape(b, 1, self.box_size, self.box_size)
        
        proj = convolve_ctf(proj, ctf)
        
        if self.invert:
            proj = - proj
            
        loss = self.distortion_loss(proj, img).unsqueeze(0)
        
        return loss

    def distortion_loss(self, x, y):
        return nn.MSELoss()(x, y)

    def get_volume(self):
        return (1 - self.mask) * self.micelle + self.volume * self.volume_scale
    
