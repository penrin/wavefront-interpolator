# coding: utf-8
'''
音場補間サンプルスクリプト
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import interpolator


if __name__ == '__main__':


    ## ----- (1) Interpolation インスタンス作成 -----

    ip = interpolator.Interpolation()


    ## ----- (2) 波数と各種座標を設定し，インスタンス変数に渡す -----

    # 波数設定
    freq = 8000 # [Hz]
    c = 340 # [m/s] sound speed
    wavelength = c / freq
    wavenumber = 2. * np.pi / wavelength # 波数
    # インスタンス変数に渡す
    ip.wavenumber = wavenumber


    # サンプル点・補間点（格子状に配置）
    nRow = 12
    nCol = 8
    width = 0.015 # [m] sample interval
    m = 7 # interpolation multiple
    ys, xs = (np.mgrid[0:nRow:1, 0:nCol:1] + 0.5) * width # サンプル点
    yi, xi = (np.mgrid[0:nRow * m:1, 0:nCol * m:1] + 0.5) * width / m # 補間点
    # インスタンス変数に渡す
    ip.xy_sample = (xs.reshape(-1), ys.reshape(-1))
    ip.xy_interpolation = (xi.reshape(-1), yi.reshape(-1))
    

    # 境界設定
    '''
    # 円形の境界（自動生成）
    center_x = np.mean(xi.reshape(-1)) # 円の中心
    center_y = np.mean(yi.reshape(-1)) 
    radius = 0.12 # [m] 半径
    NBE = 300 # 境界要素数

    ew, _ = ip.roundboundary((center_x, center_y), radius, NBE)
    '''
    # 矩形の境界（自動生成）
    center_x = np.mean(xi.reshape(-1)) # 矩形の中心
    center_y = np.mean(yi.reshape(-1)) 
    margin = 0.045 # [m] 余白（補間点・境界の最低距離）
    ratio = (18, 14) # 矩形のアスペクト比 (タテ, ヨコ) 整数。
    NBE = sum(ratio) * 2 * 5 # NBE: 境界要素数
    
    ew, _ = ip.squareboundary((center_x, center_y), ratio, margin, NBE)
    
    print('λ/d: %.2f' % (wavelength / ew))


    # レイアウト確認
    fig, ax = ip.plotlayout()
    plt.pause(0.001)



    ## ----- （デモ用に音圧分布を計算） -----

    # 円筒波源位置（境界の外側であること）
    src_x, src_y = (-0.2, -0.2)

    # オリジナル（補間点を流用）
    r_original = np.sqrt((xi - src_x) ** 2 + (yi - src_y) ** 2)
    p_original = np.cos(wavenumber * r_original) / np.sqrt(r_original)

    # サンプリング
    r_sampling = np.sqrt((xs - src_x) ** 2 + (ys - src_y) ** 2)
    p_sampling = np.cos(wavenumber * r_sampling) / np.sqrt(r_sampling)



    ## ----- (3) 補間行列の作成 -----

    IM = ip.ipmat(v=True) # v: 詳細表示



    ## ----- (4) 補間 -----

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
            vmin=-1., vmax=1., interpolation='bicubic')

    ax1 = fig.add_subplot(142)
    ax1.set_title('Sampling')
    ax1.imshow(p_sampling / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='none')

    ax2 = fig.add_subplot(143)
    ax2.set_title('Interpolation')
    ax2.imshow(p_interpolation / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='bicubic')

    ax3 = fig.add_subplot(144)
    ax3.set_title('Error')
    ax3.imshow(err / MAX, cmap=plt.cm.bwr,
            vmin=-1., vmax=1., interpolation='bicubic')

    #plt.savefig('sample.jpg')
    plt.show()

