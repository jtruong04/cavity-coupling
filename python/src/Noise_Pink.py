import numpy as np
from scipy.special import sici

from .Noise import Noise


class PinkNoise(Noise):
    def __init__(self, wl=2.0 * np.pi * 1e-12, wh=2.0 * np.pi * 1e3):
        self.wl = wl
        self.wh = wh
        super().__init__()

    @staticmethod
    def si(x):
        return sici(x)[0]

    @staticmethod
    def ci(x):
        return sici(x)[1]

    def St(self, t):
        return (self.ci(self.wh * t) - self.ci(self.wl * t)) / np.log(self.wh / self.wl)

    def Ciw(self, t, w):
        return 2.0 * (
            self.ci(np.abs(w * t + self.wh * t))
            + self.ci(np.abs(w * t - self.wh * t))
            - self.ci(np.abs(w * t + self.wl * t))
            - self.ci(np.abs(w * t - self.wl * t))
        )

    def Siw(self, t, w):
        return 2.0 * (
            self.si(w * t + self.wh * t)
            + self.si(w * t - self.wh * t)
            - self.si(w * t + self.wl * t)
            - self.si(w * t - self.wl * t)
        )

    def f0(self, t):
        return t * self.St(t) - (
            np.sin(self.wh * t) / self.wh - np.sin(self.wl * t) / self.wl
        ) / np.log(self.wh / self.wl)

    def ft(self, t):
        return (
            0.5 * t * t * self.St(t)
            - t
            * (np.sin(self.wh * t) / self.wh - np.sin(self.wl * t) / self.wl)
            / (2.0 * np.log(self.wh / self.wl))
            + (
                (1 - np.cos(self.wh * t)) / self.wh ** 2
                - (1 - np.cos(self.wl * t)) / self.wl ** 2
            )
            / (2.0 * np.log(self.wh / self.wl))
        )

    def fsin(self, t, w):
        return -np.cos(w * t) * self.St(t) / w + (
            self.Ciw(t, w)
            - 2
            * np.log(
                np.abs(
                    ((self.wh ** 2 - w ** 2) / (self.wh ** 2))
                    * ((self.wl ** 2) / (self.wl ** 2 - w ** 2))
                )
            )
        ) / (4 * w * np.log(self.wh / self.wl))

    def fcos(self, t, w):
        return np.sin(w * t) * self.St(t) / w - self.Siw(t, w) / (
            4 * w * np.log(self.wh / self.wl)
        )

    def ftsin(self, t, w):
        return (
            -self.St(t) * np.cos(w * t) * t / w
            + (
                self.St(t) * np.sin(w * t) / w
                - self.Siw(t, w) / (4 * w * np.log(self.wh / self.wl))
            )
            / w
            + (
                ((np.sin(w * t + self.wh * t)) / (w + self.wh))
                + ((np.sin(w * t - self.wh * t)) / (w - self.wh))
                - ((np.sin(w * t + self.wl * t)) / (w + self.wl))
                - ((np.sin(w * t - self.wl * t)) / (w - self.wl))
            )
            / (2 * w * np.log(self.wh / self.wl))
        )

    def ftcos(self, t, w):
        return (
            self.St(t) * np.sin(w * t) * t / w
            + (
                self.St(t) * np.cos(w * t) / w
                - (
                    self.Ciw(t, w)
                    - 2
                    * np.log(
                        np.abs(
                            ((self.wh ** 2 - w ** 2) / (self.wh ** 2))
                            * ((self.wl ** 2) / (self.wl ** 2 - w ** 2))
                        )
                    )
                )
                / (4 * w * np.log(self.wh / self.wl))
            )
            / w
            - (
                ((1 - np.cos(w * t + self.wh * t)) / (w + self.wh))
                + ((1 - np.cos(w * t - self.wh * t)) / (w - self.wh))
                - ((1 - np.cos(w * t + self.wl * t)) / (w + self.wl))
                - ((1 - np.cos(w * t - self.wl * t)) / (w - self.wl))
            )
            / (2 * w * np.log(self.wh / self.wl))
        )
