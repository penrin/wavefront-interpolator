# coding: utf-8
#--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# 音場補間サンプルスクリプト

#--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import numpy as np
import matplotlib.pyplot as plt
import ipmat


if __name__ == '__main__':

    # ----- (1) 波数と各種座標を設定 -----

    # 波数設定
    freq = 8000 # [Hz]
    c = 340 # [m/s] sound speed
    wavelength = c / freq
    wavenumber = 2. * np.pi / wavelength


    # サンプル点・補間点（格子状に配置）
    nRow = 12
    nCol = 8
    width = 0.015 # [m] sample interval
    m = 7 # interpolation multiple
    ys, xs = (np.mgrid[0:nRow:1, 0:nCol:1] + 0.5) * width # サンプル点
    yi, xi = (np.mgrid[0:nRow * m:1, 0:nCol * m:1] + 0.5) * width / m # 補間点


    # 境界設定（２点 Gauss-Legendre 求積法による離散点）
    NBE = 300
    center_x = np.mean(xi.reshape(-1))
    center_y = np.mean(yi.reshape(-1))
    distance = np.sqrt(
            (xi.reshape(-1) - center_x) ** 2 +
            (yi.reshape(-1) - center_y) ** 2
            )
    R = distance.max() * 1.2
    angle = 2. * np.pi / NBE
    print('λ/d: %.2f' % (wavelength / (R * angle)))
    theta = np.linspace(0, 2. * np.pi, NBE)
    thetaP = theta + angle / (2. * np.sqrt(3))
    thetaM = theta - angle / (2. * np.sqrt(3))
    elementP_x = R * np.cos(thetaP) + center_x
    elementP_y = R * np.sin(thetaP) + center_y
    elementM_x = R * np.cos(thetaM) + center_x
    elementM_y = R * np.sin(thetaM) + center_y
    vectorP_x = np.cos(thetaP)
    vectorP_y = np.sin(thetaP)
    vectorM_x = np.cos(thetaM)
    vectorM_y = np.sin(thetaM)
    xb = np.r_[elementM_x, elementP_x]
    yb = np.r_[elementM_y, elementP_y]
    ub = np.r_[vectorM_x, vectorP_x]
    vb = np.r_[vectorM_y, vectorP_y]



    ## ----- （デモ用に音圧分布を計算） -----

    # 音源位置（境界の外側であること）
    src_x, src_y = (-0.2, -0.2)

    # オリジナル（補間点を流用）
    r_original = np.sqrt((xi - src_x) ** 2 + (yi - src_y) ** 2)
    p_original = (1. / r_original) * np.exp(-1.j * wavenumber * r_original).real

    # サンプリング
    r_sampling = np.sqrt((xs - src_x) ** 2 + (ys - src_y) ** 2)
    p_sampling = (1. / r_sampling) * np.exp(-1.j * wavenumber * r_sampling).real



    ## ----- (2) Interpolation インスタンス作成 -----

    ip = ipmat.Interpolation()



    ## ----- (3) 波数と各種座標をインスタンス変数に渡す -----

    ip.wavenumber = wavenumber # 波数
    ip.xy_sample = (xs.reshape(-1), ys.reshape(-1)) # サンプル点
    ip.xy_interpolation = (xi.reshape(-1), yi.reshape(-1)) # 補間点
    ip.xy_boundary = ((xb, yb), (ub, vb)) # 境界

    # レイアウト確認
    fig, ax = ip.plotlayout()
    plt.pause(0.001)



    ## ----- (4) 補間行列の作成 -----

    IM = ip.ipmat(v=True) # v: 詳細表示



    ## ----- (5) 補間 -----

    # サンプル点の音圧分布の列ベクトルに
    # 補間行列を左から掛ける。

    ps = p_sampling.reshape(-1, 1) # 列ベクトル
    p_interpolation = np.dot(IM, ps).real.reshape(nRow * m, nCol * m)


    ## ----- プロット -----

    # SNR 算出
    err = p_interpolation - p_original
    SNR = 10. * np.log10(np.sum(p_original ** 2) / np.sum(err ** 2))
    print('SNR: %.1f [dB]' % SNR)

    # 描画
    MAX = max(np.max(np.abs(p_original)), np.max(np.abs(p_interpolation)))

    fig = plt.figure(figsize=(15, 5))

    ax0 = fig.add_subplot(141)
    ax0.set_title('Original')
    ax0.imshow(p_original / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='None')

    ax1 = fig.add_subplot(142)
    ax1.set_title('Sampling')
    ax1.imshow(p_sampling / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='None')

    ax2 = fig.add_subplot(143)
    ax2.set_title('Interpolation')
    ax2.imshow(p_interpolation / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='None')

    ax3 = fig.add_subplot(144)
    ax3.set_title('Error')
    ax3.imshow(err / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='None')

    #plt.savefig('sample.jpg')
    plt.show()

