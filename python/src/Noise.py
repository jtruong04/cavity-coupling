import numpy as np


class Noise:
    def __init__(self):
        pass

    def __call__(self, t, w1, w2):
        return self.J(t, w1, w2)

    # ==============================================================
    # 
    # The following functions need to be overloaded when inherited
    # 
    # Simply define S(w). Normalize it so that S(t=0) = 1. Then
    # compute the following integrals whether that be numerical
    # or analytical
    # 
    # ==============================================================
    
    def Sw(self, w):
        '''Power spectral density function'''
        return 0

    def St(self, t):
        '''Temporal correlation function
        
            S(t) = \int_{-\infty}^{\infty} S(w) exp(-iwt) dw
        '''
        return 0

    def f0(self, t):
        '''
            \int_0^{t} dt' S(t')
        '''
        return 0

    def ft(self, t):
        '''
            \int_0^{t} dt' S(t') * t'
        '''
        return 0

    def fsin(self, t, w):
        '''
            \int_0^{t} dt' S(t') * sin(wt')
        '''
        return 0

    def fcos(self, t, w):
        '''
            \int_0^{t} dt' S(t') * cos(wt')
        '''
        return 0

    def ftsin(self, t, w):
        '''
            \int_0^{t} dt' S(t') * t' sin(wt')
        '''
        return 0

    def ftcos(self, t, w):
        '''
            \int_0^{t} dt' S(t') * t' cos(wt')
        '''
        return 0
    # 
    # ============================================================
    #
     
    def JS(self, t, w1, w2):
        if t == 0:
            return 0
        if w1 == 0 and w2 == 0:
            return -(self.ft(t) - t * self.f0(t))
        elif w1 == -w2 and w2 != 0:
            return -(self.ftcos(t, w1) - t * self.fcos(t, w1))
        elif w1 != 0 and w2 == 0:
            return -(np.exp(1j * w1 * t / 2) / w1) * (
                np.cos(w1 * t / 2) * self.fsin(t, w1)
                - np.sin(w1 * t / 2) * (self.fcos(t, w1) + self.f0(t))
            )
        elif w1 == 0 and w2 != 0:
            return -(np.exp(1j * w2 * t / 2) / w2) * (
                np.cos(w2 * t / 2) * self.fsin(t, w2)
                - np.sin(w2 * t / 2) * (self.fcos(t, w2) + self.f0(t))
            )
        else:
            return -(np.exp(1j * (w1 + w2) * t / 2) / (w1 + w2)) * (
                np.cos((w1 + w2) * t / 2) *
                (self.fsin(t, w1) + self.fsin(t, w2))
                - np.sin((w1 + w2) * t / 2) *
                (self.fcos(t, w1) + self.fcos(t, w2))
            )

    def JA(self, t, w1, w2):
        if t == 0:
            return 0
        if w1 == 0 and w2 == 0:
            return 0
        elif w1 == -w2 and w2 != 0:
            return -1j * (self.ftsin(t, w1) - t * self.fsin(t, w1))
        elif w1 != 0 and w2 == 0:
            return (
                -1j
                * (np.exp(1j * w1 * t / 2) / w1)
                * (
                    -np.sin(w1 * t / 2) * self.fsin(t, w1)
                    - np.cos(w1 * t / 2) * (self.fcos(t, w1) - self.f0(t))
                )
            )
        elif w1 == 0 and w2 != 0:
            return (
                -1j
                * (np.exp(1j * w2 * t / 2) / w2)
                * (
                    np.sin(w2 * t / 2) * self.fsin(t, w2)
                    + np.cos(w2 * t / 2) * (self.fcos(t, w2) - self.f0(t))
                )
            )
        else:
            return (
                -1j
                * (np.exp(1j * (w1 + w2) * t / 2) / (w1 + w2))
                * (
                    np.cos((w1 + w2) * t / 2) *
                    (self.fcos(t, w1) - self.fcos(t, w2))
                    + np.sin((w1 + w2) * t / 2) *
                    (self.fsin(t, w1) - self.fsin(t, w2))
                )
            )

    def J(self, t, w1, w2):
        if t <= 0:
            return 0
        return self.JS(t, w1, w2) + self.JA(t, w1, w2)
