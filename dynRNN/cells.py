from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import torch


class DynCell(ABC):
    @abstractmethod
    def update(self, h_t, s_t, pars, x=0):
        pass

    def phaseplane(self, x_grid, y_grid, pars, input, bifurcation=False):
        # return fig ?        
        x_next = self.update(y_grid, x_grid, pars, input)

        f, ax = plt.subplots()
        if bifurcation:
            ax.contourf(x_grid, y_grid, x_next - y_grid, 0)
        else:
            ax.contourf(x_grid, y_grid, x_next[0] - y_grid, 0)
            ax.contour(x_grid, y_grid, x_next[1] - x_grid, 0, colors="red")
            # ax.quiver(x_grid, y_grid, x_next[1] - x_grid, x_next[0] - y_grid)

        return ax

    def trajectory(self, h0, s0, pars, x_t):
        h_l = [h0]
        s_l = [s0]

        for x in x_t:
            next_hs = self.update(h_l[-1], s_l[-1], pars, x)
            if type(next_hs) is not tuple:
                h_l.append(next_hs)
            else:
                h_l.append(next_hs[0])
                s_l.append(next_hs[1])

        return h_l, s_l
    

    # def bifurcation(self, x_grid, y_grid, pars, bif_par, input, tol = 1e-2): #bif_par of the form (idx, [list of values])
    #     f, ax = plt.subplots()

    #     pars = list(pars)
    #     for p in bif_par[1]:
    #         pars[bif_par[0]] = p
        
    #         x_next = self.update(y_grid, x_grid, pars, input)

    #         #DEAL WITH 2D CASE
    #         fp_eval = (x_next[0] - y_grid).abs() + (x_next[1] - x_grid).abs()
    #         mask = fp_eval < tol
    #         ax.plot(torch.ones(mask.sum())*p, x_next[0][mask].flatten(), '.', markersize = 3)

    #     return ax


class BRCell(DynCell):
    def update(self, h_t, s_t, pars, x=0):
        Pa, C, W, Wc, Wa = pars
        c = torch.sigmoid(C*h_t + Wc*x)
        a = 1 + torch.tanh(Pa*h_t + s_t + Wa*x)
        h_next = (1 - c) * h_t + c * torch.tanh(W*x + a * h_t)

        return h_next


class LSTMCell(DynCell):
    def update(self, h_t, s_t, pars, x=0):
        A, B, C, D, P, W, \
        Wa, Wb, Wd, Wp = pars
        a = torch.sigmoid(Wa*x + A*h_t)
        b = torch.sigmoid(Wb*x + B*h_t)
        d = torch.sigmoid(Wd*x + D*h_t)

        s_next = a * s_t + b * torch.tanh(W * x + C * h_t)
        h_next = d * torch.tanh(s_next) + P*h_t

        return h_next, s_next


class TRCell(DynCell):
    def update(self, h_t, s_t, pars, x=0):
        B, C = pars
        h_next = (1 - C) * h_t + C * torch.tanh(x + s_t * h_t + B * h_t**3)

        return h_next


class FlexCell(DynCell):
    def update(self, h_t, s_t, pars, x=0):
        A, B, C, D, E = pars
        h_next = (1 - C) * h_t + C * torch.tanh(x + (A + B * h_t**2 - s_t) * h_t)
        s_next = (1 - D) * s_t + D * (E * h_t) ** 4

        return h_next, s_next
