"""
Lei code:
bispectrum and bicoherence in accumulated way to get 2d complex array of [f,f] to save memory
"""

import numpy as np
import pywt
# print(pywt.__version__)

# system properties
surf_ten = 0.0728   # surface tension [N/m]
density = 998       # fluid density [kg/m**3]
h = 725/10**6       # fluid volume (or basin) depth [m]
L = 9.525/10**3     # fluid volume (or basin) width [m]
km = np.pi/L        # smallest possible wavemode wavenumber given homogeneous boundary [rad/m]

def wav_bispectrum(zf, t, fs, fb_low, fb_high, scales=None, Ns=2**10, wavelet='cgau1', bico=False):
    """
    Wavelet-based bispectrum.

    Returns
    -------
    f_out : 1d float array  
    Bi_spec_avg: 2d complex array, <B(f1,f2)>_t(time average)
    """

    # scales grid
    if scales is None:
        scales = 34500/np.linspace(fb_high, fb_low, Ns)

    # wavelet transform
    spec, f = pywt.cwt(zf, scales, wavelet, 1/fs)

    # arguments to spectra, flip f to ascending
    idx_in = np.arange(f.size//2)
    idx_sum = idx_in[:, None] + idx_in[None,:]
    f_out = f[::-1][idx_in]

    # transpose (f,t) -> (t,f), flip for ascending in f
    spec = np.transpose(spec, [1,0])[:,::-1].astype('complex64')
                        
    acc = np.zeros((idx_in.size, idx_in.size), dtype=spec.dtype)
    for k in range(spec.shape[0]):  
        s = spec[k]                 # (k,f)
        s_in = s[idx_in]            # (k,f/2)
        bi_t = s_in[:, None] * s_in[None, :] * np.conjugate(s[idx_sum])
        acc += bi_t
    
    Bi_spec_avg = acc / spec.shape[0] 

    # crop to relevant spectrum
    if not bico:
        f_idx = (f_out>=fb_low) & (f_out<=fb_high)
        f_out = f_out[f_idx]
        Bi_spec_avg = Bi_spec_avg[f_idx,:]
        Bi_spec_avg = Bi_spec_avg[:,f_idx]

    return f_out, Bi_spec_avg   


def wav_bicoherence(zf, t, fs, fb_low, fb_high, scales=None, Ns=2**10, wavelet='cgau1'):
    """
    Wavelet-based bicoherence.
    """

    # scales grid
    if scales is None:
        scales = 34500/np.linspace(fb_high,fb_low,Ns)

    # wavelet transform
    spec, f = pywt.cwt(zf, scales, wavelet, 1/fs)

    # arguments to spectra, flip f to ascending
    idx_in = np.arange(f.size//2)
    idx_sum = idx_in[:, None] + idx_in[None,:]
    f_out = f[::-1][idx_in]

    # transpose (f,t) -> (t,f), flip for ascending in f
    spec = np.transpose(spec, [1,0])[:, ::-1].astype('complex64')

    num_acc = np.zeros((idx_in.size, idx_in.size), dtype=spec.dtype)  # <B(f1,f2)>
    den_acc = np.zeros((idx_in.size, idx_in.size), dtype=np.float32)  # <|B(f1,f2)|>
    
    T = spec.shape[0]
    for k in range(spec.shape[0]):
        s = spec[k]
        bi_t = (s[idx_in, None] * s[None, idx_in] * np.conjugate(s[idx_sum])).astype('complex64')
        num_acc += bi_t
        den_acc += np.abs(bi_t)

    # bicoherence = |<B>| / <|B|>
    num = np.abs(num_acc / T)
    den = den_acc / T
    Bi_co = num / den

    f_idx = (f_out>=fb_low) & (f_out<=fb_high)
    f_out = f_out[f_idx]
    num = num[f_idx,:]
    num = num[:,f_idx]
    Bi_co = Bi_co[f_idx,:]
    Bi_co = Bi_co[:,f_idx]

    return f_out, num, Bi_co


