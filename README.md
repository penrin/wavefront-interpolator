Wavefront Interpolator
======================


要件
----

* Python 3
* Numpy, Scipy, Matplotlib


概要
----

* 音場の離散点測定データからの連続分布の復元・補間
* Kirchhoff-Helmholtz 積分方程式に基づき，逆問題で音場内のサンプル点から境界ポテンシャルを推定し，順問題で内部の音場を計算。２次元音場に対応。


参考文献
--------

* 矢田部浩平, 及川靖広, ``レーザドプラ振動計を用いた音場測定への境界要素法の逆解析の導入,'' 電子情報通信学会論文誌 A Vol.J97-A No.2 pp.104-111 (2014)


インストール
-----------

```
$ git clone https://github.com/penrin/wavefront-interpolator
```

サンプルスクリプト実行
-------------------

```
$ cd wavefront-interpolator
$ python sample.py
```