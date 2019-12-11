MXEライブラリ libmxe
========

MRI.COMに関する基本的なサブルーチンをまとめたライブラリ。
実験の前処理、後処理、解析で利用されることを想定。


内容
--------

### 独立したライブラリ

* libmxe_calendar: カレンダー


### MRI.COM実験の設定情報を読み込むモジュール群

複数の実験設定を扱えるようにクラス、インスタンスの概念を導入している。

* libmxe_para: 実験の基本パラメータ
* libmxe_grid: グリッド
* libmxe_topo: 海底地形
* libmxe_io:   データの入出力(レコードの時間)


### 上記のモジュール群を利用した単機能モジュール

* libmxe_grads: gradsコントロール・ファイルを作成する


使用方法
--------

ユーザーは事前に、[MXEライブラリ用namelist](http://synthesis.jamstec.go.jp/heroes/docs/html/mri.com-user_mxe_lib_namelist.html)を作成しておきます。

tripolar以外の一般座標変換を用いるには
1. MRI.COMの座標変換プログラムを一部修正してlib/trnsfrmに置き
2. libmxe_trnsfrm.F90 からリンクを張る
どのような修正が必要かは MRI.COMのtrnsfrm/trnsfrm.tripolar.F90とのdiffを見るといい。

