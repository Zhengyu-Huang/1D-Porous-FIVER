import numpy as np

class Structure:
    def __init__(self,ms, ks, alpha, L, mode):
        '''
        :param ms: structure mass
        :param ks: structure stiffness, if the structure is modeled as piston-spring system  $$ms \dot\dot{xs} + ks(xs - xs_0) = f(t) = p_B$$
        :param alpha: poros ratio, 0 means nonpenetrable, 1 means fully-penetrable
        :param L: tube length, the fluid domain is [0,L]

        Attributes
        qq_h: structure current position and velocity at time n-0.5
        qq_n: predicted structure position and velocity at time n
        qq_old_n: predicted structure position and velocity at time n-1(the predicted values from last step)
        t: current time
        mode: structure motion type.  0: piston-spring system ;
                                      1: forced motion, advancing type 1
                                      2: forced motion, advancing type 2
                                      3: forced motion, harmonic motion
        '''
        self.ms = ms
        self.ks = ks
        self.qq_n = np.array([L/2.0, 0.0])
        self.qq_old_n = np.array([L/2.0, 0.0])
        self.qq_h = np.array([L/2.0, 0.0])
        self.L = L





        self.t = 0
        self.porous_ratio = alpha

        self.mode = mode

    def _forced_motion(self, t,mode):
        # self.t = t
        if mode == 1:
            xs = self.L / 2 + 1.0 / 8.0 * t ** 4
            vs = 1.0 / 2.0 * t ** 3

        elif mode == 2:
            xs = self.L / 2.0 - 0.25 * t
            vs = -0.25


        elif mode == 3:
            u0 = 0.1;
            w = 2.0;
            xs = self.L / 2.0 + u0 * np.cos(w * t) - u0
            vs = -u0 * w * np.sin(w * t)

        elif mode == 4:
            xs = self.L/2.0
            vs = 0

        # self.qq_n = np.array([xs,vs, as_0])
        return np.array([xs, vs])



    def _move(self, f, dt):

        mode = self.mode

        if(mode == 0):

            [xs_old_n, vs_old_n] = self.qq_h

            as_ = (f - self.ks * (xs_old_n - self.L / 2 + dt * vs_old_n / 2)) / (
            self.ms + dt * dt * self.ks / 4)

            self.qq_old_n = self.qq_n

            vs = vs_old_n + dt * as_

            xs = xs_old_n + dt * (vs_old_n + vs) / 2

            self.qq_h = np.array([xs, vs])

            self.qq_n = np.array([xs + 0.5 * dt * vs + 0.125 * dt * (vs - vs_old_n), 1.5 * vs - 0.5 * vs_old_n])
            # Because we do not use acceleration, set it as default value 0


        # for forced motion, we do not need qq_h
        else:
            self.qq_old_n = self.qq_n

            self.qq_n = self._forced_motion(self.t + dt, mode)

        self.t = self.t + dt

    def _predict_half_step(self, f, dt):

        mode = self.mode

        if mode == 0:

            [xs_0, vs_0] = self.qq_n

            as_0 = f / self.ms

            # there just predict the qq_{1/2}

            xs = xs_0 + dt * vs_0 + dt * dt * as_0 / 2.0

            vs = vs_0 + dt * as_0

            self.qq_n = np.array([xs, vs])

            self.qq_old_n = np.array([xs_0, vs_0])

            self.qq_h = np.array([xs_0, vs_0])
        else:

            self.qq_old_n = self.qq_n

            self.qq_h = self.qq_n

            self.qq_n = self._forced_motion(self.t + dt, mode)
