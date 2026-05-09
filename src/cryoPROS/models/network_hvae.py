import torch
import torch.nn as nn
import torch.nn.init as init
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
           box_size // 4 : 3 * box_size // 4] = 1
    
    return volume

def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ProjNet(nn.Module):
    def __init__(self, nf=64, num_down=0):
        super(ProjNet, self).__init__()
        
        conv = []
        
        conv.append(nn.Conv2d(1, nf, 3, 1, 1))
        for _ in range(num_down):
            conv.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            conv.append(nn.Conv2d(nf, nf, 4, 2, 1))
        
        conv.append(nn.Conv2d(nf, nf, 3, 1, 1))
        self.conv = nn.Sequential(*conv)
        
    def forward(self, proj):
        proj = self.conv(proj)
        
        return proj


class MetaNet(nn.Module):
    def __init__(self, in_dim=6, nf=64, base_h=16, base_w=16, num_up=0):
        super(MetaNet, self).__init__()
        
        mlp_out_dim = base_h*base_w        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_out_dim//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Linear(mlp_out_dim//4, mlp_out_dim//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Linear(mlp_out_dim//2, mlp_out_dim)
            )
        
        conv = []
        
        conv.append(nn.Conv2d(1, nf, 3, 1, 1))
        for _ in range(num_up):
            conv.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            conv.append(nn.ConvTranspose2d(nf, nf, 4, 2, 1))
        
        conv.append(nn.Conv2d(nf, nf, 3, 1, 1))
        
        self.conv = nn.Sequential(*conv)        
        self.base_h = base_h
        self.base_w = base_w
        
    def forward(self, angle):
        b = angle.shape[0]
        
        proj = self.mlp(angle)
        proj = proj.reshape(b, 1, self.base_h, self.base_w)
        proj = self.conv(proj)
        
        return proj

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.net = ResidualDenseBlock_5C(in_channels)
        
    def forward(self, x):
        return self.net(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.down = nn.Sequential(
                    nn.PixelUnshuffle(2), 
                    nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1), 
                    )
        
    def forward(self, x):
        spatial_size = x.shape[-1]
        if spatial_size % 2 == 1:
            x = F.interpolate(x, size=(spatial_size + 1, spatial_size + 1))
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
                    nn.PixelShuffle(2), 
                    nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1)
                    )
        
    def forward(self, x):
        return self.up(x)

class GauBlock(nn.Module):
    def __init__(self, in_channels, out_channels, zero_last=True):
        super().__init__()
        
        self.m = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 1)
                 )
        
        self.v = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 1)
                 )
        
        if zero_last:
            self.m[1].weight.data *= 0
            self.v[1].weight.data *= 0
        
    def forward(self, x):
        return self.m(x), self.v(x)


def degrade_process(proj, noise_level):
    noise = torch.randn_like(proj)
    noise = noise / torch.norm(noise, dim=(2, 3), keepdim=True)
    noise = noise_level * torch.norm(proj, dim=(2, 3), keepdim=True) * noise
    
    proj = proj + noise
    
    return proj


class HVAE(nn.Module):
    def __init__(self, in_ch=1, nf=64, nls=[2, 2], z_dim=16, box_size=256, 
                 Apix=1.0, invert=True, init_volume_path=None, update_volume=False, 
                 volume_scale=1.0, noise_level=0):
        super(HVAE, self).__init__()
        
        self.inconv = nn.Conv2d(in_ch, nf, 1)

        num_down = len(nls) - 1
        base_h = int(box_size // (2 ** num_down))
        base_w = int(box_size // (2 ** num_down))

        encs = []
        Gauconv_q = []
        Gauconv_p = []
        decs = []
        proj_z = []
        proj_nets = []
        meta_nets = []
        spatial_sizes = []
        
        for i, nl in enumerate(nls):

            for _ in range(nl):
                encs.append(BasicBlock(nf))
                Gauconv_q.append(GauBlock(nf, z_dim))
                Gauconv_p.append(GauBlock(nf, z_dim))
                decs.append(BasicBlock(nf))
                proj_z.append(nn.Conv2d(z_dim, nf, 1))
                proj_nets.append(ProjNet(nf, i))
                meta_nets.append(MetaNet(6, nf, base_h, base_w, num_down-i))

                spatial_size = int(box_size // (2 ** i)) if box_size % (2 ** i) == 0 else int(box_size // (2 ** i)) + 1
                spatial_sizes.append(spatial_size)
                
            if i != len(nls) - 1:
                Gauconv_q.append(GauBlock(nf, z_dim))
                Gauconv_p.append(GauBlock(nf, z_dim))
                encs.append(DownSample(nf, nf))
                decs.append(UpSample(nf, nf))
                proj_z.append(nn.Conv2d(z_dim, nf, 1))
                proj_nets.append(ProjNet(nf, i+1))
                meta_nets.append(MetaNet(6, nf, base_h, base_w, num_down-i-1))

                spatial_size = int(box_size // (2 ** (i+1))) if box_size % (2 ** (i+1)) == 0 else int(box_size // (2 ** (i+1))) + 1
                spatial_sizes.append(spatial_size) 
        
        self.encs = nn.ModuleList(encs)
        self.Gauconv_q = nn.ModuleList(Gauconv_q)
        self.Gauconv_p = nn.ModuleList(Gauconv_p)
        self.decs = nn.ModuleList(decs)
        self.proj_z = nn.ModuleList(proj_z)
        self.proj_nets = nn.ModuleList(proj_nets)
        self.meta_nets = nn.ModuleList(meta_nets)

        self.spatial_sizes = spatial_sizes
        
        self.outconv = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), 
                                     nn.ReLU(True), 
                                     nn.Conv2d(nf, in_ch, 3, 1, 1))
        
        num_down = len(nls) - 1
        size_h = int(box_size / (2 ** num_down))
        size_w = int(box_size / (2 ** num_down))
        
        self.dec0 = nn.Parameter(torch.ones(1, nf, size_h, size_w))
        self.nls = nls
                
        if init_volume_path is not None:
            with mrcfile.open(init_volume_path, permissive=True) as mrc:
                volume = mrc.data
            volume = torch.from_numpy(volume.astype(np.float32))
        else:
            volume = init_volume(box_size)
        
        volume = volume_scale * (volume / torch.norm(volume))
        
        if update_volume:
            self.volume = nn.Parameter(volume, requires_grad=True)
        else:
            self.volume = nn.Parameter(volume, requires_grad=False)
        
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
        self.freqs = nn.Parameter(torch.from_numpy(freqs).unsqueeze(0).float(), requires_grad=False)
        self.box_size = box_size
        
        self.invert = invert
        self.noise_level = noise_level


    def forward(self, img, rotation, trans, ctf_para, meta):
        b, c, h, w = img.shape
    
        acts = self.encode(img)
        proj = self.forward_model(rotation, trans, ctf_para)
        proj_degraded = degrade_process(proj, self.noise_level)
        
        model_loss = self.distortion_loss(proj, img)
        
        dec, kl_loss = self.decode(acts, proj_degraded, meta)
        rec_loss =  self.distortion_loss(dec, img)
        
        kl_loss = kl_loss / (b*c*h*w)
        
        kl_loss = kl_loss.unsqueeze(0)
        rec_loss = rec_loss.unsqueeze(0)
        model_loss = model_loss.unsqueeze(0)
        
        return rec_loss, kl_loss, model_loss


    def encode(self, img):
        conv_in = self.inconv(img)
        
        acts = []
        act = conv_in
        
        for enc in self.encs:
            act = enc(act)
            acts.append(act)
            
        return acts


    def decode(self, acts, proj, meta):
        kls = 0
        dec = self.dec0.repeat(acts[0].shape[0], 1, 1, 1)
        
        for i in range(len(self.decs))[::-1]:
            
            act = acts[i]

            spatial_size = self.spatial_sizes[i]

            proj_emd = self.proj_nets[i](proj)
            meta_emd = self.meta_nets[i](meta)

            dec = F.interpolate(dec, size=(spatial_size, spatial_size))
            proj_emd = F.interpolate(proj_emd, size=(spatial_size, spatial_size))
            meta_emd = F.interpolate(meta_emd, size=(spatial_size, spatial_size))
            
            qm, qv = self.Gauconv_q[i](dec + act + proj_emd + meta_emd)
            pm, pv = self.Gauconv_p[i](dec + proj_emd + meta_emd)
            
            enc = draw_gaussian_diag_samples(qm, qv)
            kl = gaussian_analytical_kl(qm, pm, qv, pv)
            
            dec = dec + self.proj_z[i](enc)
            
            dec = self.decs[i](dec)
            kls = kls + kl.sum()
            
        dec = self.outconv(dec)
        
        return dec, kls


    def decode_uncond(self, proj, meta, temperature=1.0):
        bs = proj.shape[0]
        dec = self.dec0.repeat(bs, 1, 1, 1)
        
        for i in range(len(self.decs))[::-1]:

            spatial_size = self.spatial_sizes[i]

            proj_emd = self.proj_nets[i](proj)
            meta_emd = self.meta_nets[i](meta)

            dec = F.interpolate(dec, size=(spatial_size, spatial_size))
            proj_emd = F.interpolate(proj_emd, size=(spatial_size, spatial_size))
            meta_emd = F.interpolate(meta_emd, size=(spatial_size, spatial_size))

            pm, pv = self.Gauconv_p[i](dec + proj_emd + meta_emd)
            enc = draw_gaussian_diag_samples(pm, pv) * temperature
            dec = dec + self.proj_z[i](enc)
            dec = self.decs[i](dec)
            
        dec = self.outconv(dec)
        
        return dec

    def distortion_loss(self, x, y):
        return nn.MSELoss()(x, y)

    def generate(self, rotation, trans, ctf_para, meta, temperature=1.0):
        proj = self.forward_model(rotation, trans, ctf_para)
        proj_degraded = degrade_process(proj, self.noise_level)
        dec = self.decode_uncond(proj_degraded, meta, temperature)
        
        return dec

    def forward_model(self, rotation, trans, ctf_para):
        b = rotation.shape[0]
        volume = self.volume.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        proj = discrete_radon_transform_3d(volume, rotation)
        proj = translation_2d(proj, 2 * trans)
        
        freqs = self.freqs.repeat(b, 1, 1)
        ctf = compute_ctf(freqs, *torch.split(ctf_para, 1, 1))
        ctf = ctf.reshape(b, 1, self.box_size, self.box_size)
        
        proj = convolve_ctf(proj, ctf)
        
        if self.invert:
            proj = - proj
            
        return proj
    
    def generate_proj(self, rotation, trans, ctf_para):
        proj = self.forward_model(rotation, trans, ctf_para)
        
        return proj



# model = HVAE(nls=[2,2,2,2], box_size=147)
# img = torch.randn(2, 1, 147, 147)
# rotation = torch.randn(2, 3, 3)
# trans = torch.randn(2, 2)
# ctf_para = torch.randn(2, 7)
# meta = torch.randn(2, 6)
# rec_loss, kl_loss, model_loss = model(img, rotation, trans, ctf_para, meta)
# dec = model.generate(rotation, trans, ctf_para, meta)
# print(dec.shape)
