# coding: utf-8
''' 
Wavefront interpolator

2017 penrin

'''

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1


class Interpolation():
    
    # 波数
    wavenumber = 0.

    # サンプル点座標
    xy_sample = (np.array([]), np.array([]))

    # 補間点座標
    xy_interpolation = (np.array([]), np.array([]))

    # 境界積分点座標（２点 Gauss-Legendre 求積用）
    xy_boundary = (
            (np.array([]), np.array([])), # 座標 (x, y)
            (np.array([]), np.array([]))  # 外向き法線ベクトル (u, v)
            )

    # 正則化パラメータ
    beta = .1
    
    
    # 各座標のレイアウト確認
    def plotlayout(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        # 補間点
        ax.plot(self.xy_interpolation[0], self.xy_interpolation[1],
                '.', color='skyblue', label='Interpolation points')
        n = [1, 2, 3, '…']
        for i, txt in enumerate(n):
            ax.annotate(
                txt, (self.xy_interpolation[0][i], self.xy_interpolation[1][i]),
                color='skyblue', ha='left', va='bottom')
        # サンプル点
        ax.plot(self.xy_sample[0], self.xy_sample[1],
                '.', color='black', label='Sampling points')
        n = [1, 2, 3, '…']
        for i, txt in enumerate(n):
            ax.annotate(txt, (self.xy_sample[0][i], self.xy_sample[1][i]),
                    ha='left', va='bottom')
        # 境界
        b_xy, b_uv = self.xy_boundary
        ax.plot(b_xy[0], b_xy[1], '.', color='salmon', label='Boundary')
        ax.quiver(b_xy[0], b_xy[1], 0.5 * b_uv[0], 0.5 * b_uv[1],
                  color='salmon', angles='xy', width=0.002)
        # options
        #ax.xaxis.tick_top()
        ax.invert_yaxis()
        ax.axis('equal')
        ax.legend()
        ax.grid()
        ax.set_title('Layout')
        return fig, ax
    
    # check
    def checkparams(self):
        errflg = 0
        if self.wavenumber == 0:
            print('warning: wavenumber is 0')
            

    # 補間行列計算
    def ipmat(self, v=False):
        if v == True:
            self.checkparams()
        mat_s2b = self.s2b(v) # inverse problem
        mat_b2i = self.b2i(v) # direct problem
        IM = np.dot(mat_b2i, mat_s2b) # interpolation matrix
        return IM


    # samples -> boundary matrix
    # (Inverse problem)
    def s2b(self, v=False):
        xs = self.xy_sample[0].reshape(-1, 1) # 列ベクトル
        ys = self.xy_sample[1].reshape(-1, 1)
        xb = self.xy_boundary[0][0].reshape(1, -1) # 行ベクトル
        yb = self.xy_boundary[0][1].reshape(1, -1)
        ub = self.xy_boundary[1][0].reshape(1, -1)
        vb = self.xy_boundary[1][1].reshape(1, -1)
        k = self.wavenumber
        G = np.c_[
                green2d(xs, ys, xb, yb, k),
                -1. * green2d_dash(xs, ys, xb, yb, ub, vb, k)
                ]
        # Psuedo-inverse matrix (Tikhonov regularization)
        beta = self.beta
        I = np.identity(G.shape[1])
        Ginv = np.dot(
                np.linalg.inv(np.dot(np.conj(G.T), G) + beta * I),
                np.conj(G.T)
                )
        return Ginv


    # boundary -> interpolation-samples matrix
    # (Direct problem)
    def b2i(self, v=False):
        xi = self.xy_interpolation[0].reshape(-1, 1) # 列ベクトル
        yi = self.xy_interpolation[1].reshape(-1, 1)
        xb = self.xy_boundary[0][0].reshape(1, -1) # 行ベクトル
        yb = self.xy_boundary[0][1].reshape(1, -1)
        ub = self.xy_boundary[1][0].reshape(1, -1)
        vb = self.xy_boundary[1][1].reshape(1, -1)
        k = self.wavenumber
        G = np.c_[
                green2d(xi, yi, xb, yb, k),
                -1. * green2d_dash(xi, yi, xb, yb, ub, vb, k)
                ]
        return G
    

    # 境界積分点（円形）を自動生成
    def roundboundary(self, center, radius, NBE):
        # center: 円の中心
        # radius: 円の半径
        # NBE: 境界要素数
        center_x = center[0]
        center_y = center[1]
        R = radius
        angle = 2. * np.pi / NBE
        ew = R * angle # Element width
        theta = np.linspace(0, 2. * np.pi, NBE)
        thetaP = theta + angle / (2. * np.sqrt(3))
        thetaM = theta - angle / (2. * np.sqrt(3))
        xb = np.r_[R * np.cos(thetaM) + center_x, R * np.cos(thetaP) + center_x]
        yb = np.r_[R * np.sin(thetaM) + center_y, R * np.sin(thetaP) + center_y]
        ub = np.r_[np.cos(thetaM), np.cos(thetaP)]
        vb = np.r_[np.sin(thetaM), np.sin(thetaP)]
        self.xy_boundary = ((xb, yb), (ub, vb))
        return ew


    # 境界積分点（矩形）の自動生成
    def squareboundary(self, center, ratio, margin, NBE):
        # center: 矩形の中心
        # ratio: 矩形のアスペクト比 (タテ, ヨコ) 整数。
        # margin: [m] 余白（補間点・境界の最低距離）
        # NBE: 境界要素数
        xi = self.xy_interpolation[0]
        yi = self.xy_interpolation[1]
        xfar = np.max(np.abs(xi - center[0]))
        yfar = np.max(np.abs(yi - center[1]))
        if (ratio[0] / ratio[1]) > (yfar / xfar):
            # x 方向の margin をとる
            xwidth = (xfar + margin) * 2
            ywidth = xwidth / ratio[1] * ratio[0]
            i = np.where(np.abs(xi - center[0]) == xfar)
        else:
            # y 方向の margin をとる
            ywidth = (yfar + margin) * 2
            xwidth = ywidth / ratio[0] * ratio[1]
            i = np.where(np.abs(yi - center[1]) == yfar)
        
        xNBE = int(NBE / 2 / sum(ratio) * ratio[1])
        yNBE = int(NBE / 2 / sum(ratio) * ratio[0])
        ew = xwidth / xNBE # Element width
        shift_m = ew * (1 - 1 /np.sqrt(3)) / 2
        shift_p = ew * (1 + 1 /np.sqrt(3)) / 2
        xb = np.r_[
                np.linspace(0, xwidth, xNBE, endpoint=False) + shift_m,
                np.linspace(0, xwidth, xNBE, endpoint=False) + shift_p,
                np.linspace(0, xwidth, xNBE, endpoint=False) + shift_m,
                np.linspace(0, xwidth, xNBE, endpoint=False) + shift_p,
                np.zeros(yNBE),
                np.zeros(yNBE),
                np.zeros(yNBE) + xwidth,
                np.zeros(yNBE) + xwidth
                ] - xwidth / 2 + center[0]
        yb = np.r_[
                np.zeros(xNBE),
                np.zeros(xNBE),
                np.zeros(xNBE) + ywidth,
                np.zeros(xNBE) + ywidth,
                np.linspace(0, ywidth, yNBE, endpoint=False) + shift_m,
                np.linspace(0, ywidth, yNBE, endpoint=False) + shift_p,
                np.linspace(0, ywidth, yNBE, endpoint=False) + shift_m,
                np.linspace(0, ywidth, yNBE, endpoint=False) + shift_p
                ] - ywidth / 2 + center[1]
        ub = np.r_[
                np.zeros(xNBE * 2),
                np.zeros(xNBE * 2),
                np.ones(yNBE * 2) * -1,
                np.ones(yNBE * 2),
                ]
        vb = np.r_[
                np.ones(xNBE * 2) * -1,
                np.ones(xNBE * 2),
                np.zeros(yNBE * 2),
                np.zeros(yNBE * 2)
                ]
        self.xy_boundary = ((xb, yb), (ub, vb))
        return ew



# 2次元グリーン関数
def green2d(xs, ys, xb, yb, k):
    # (xs, ys): Sampling Point
    # (xb, yb): Boundary Element
    # k: Wavenumber
    r = np.sqrt((xs - xb) ** 2 + (ys - yb) ** 2)
    G = (1.j / 4.) * hankel1(0., k * r)
    return G


# 2次元グリーン関数法線方向微分
def green2d_dash(xs, ys, xb, yb, ub, vb, k):
    # (xs, ys): Sampling Point
    # (xb, yb): Boundary Element
    # k: Wavenumber
    # (ub, vb): Normal Vector
    r = np.sqrt((xs - xb) ** 2 + (ys - yb) ** 2)
    innerProduct = (xs - xb) * ub + (ys - yb) * vb
    Gd = (1.j * k / 4.) * hankel1(1., k * r) * innerProduct / r
    return Gd



