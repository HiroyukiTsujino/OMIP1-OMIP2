MXE 解析ツール (anl/)
========

MRI.COM出力の解析ツール類。


各ディレクトリの内容
--------

各ディレクトリはおおよそ次のような内容で構成される。

* README-MXE.md
* Makefile
* src/    - Fortranソースコード
            テスト自動化のために一部はモジュール化されている
* sample/ - 実際にプログラムを動かすスクリプトのサンプル
* test/   - 単体テスト用プログラム (test_all.shを実行)


内容
--------

### 設定・外力

* force       - フォーシングファイルに関するツール(補間など)
* topo        - 地形に関するツール(T点の地形作成,海底直上2次元分布)
* fill_land   - 陸地を海グリッドの値で埋める
* format_conv - フォーマット変換( grads<=>netCDF, real<=>dble など)

### 検証

* verify      - history 出力のチェック (最小値最大値)

### 積分・平均

* integ       - history データの空間積分、時間平均、2つの差
* interpolate - 内挿、補間
* transport   - 流量, 輸送量, 流速の計算
* resize      - ヒストリーデータの切り出し
* cut_out     - ヒストリーデータの切り出し
* ntrans_heat - 北向き熱輸送量の計算
* ntrans_salt - 北向き塩分輸送量の計算
* moc_depth   - 子午面流線関数の計算
* gmmoc_depth - Bolus速度による子午面流線関数の計算

### 密度

* density     - 密度の計算
* potential_density - ポテンシャル密度の計算
* ip_dep      - 等密度面の深度
* iptw_trc    - トレーサーを層厚重みつき平均によって等密度面の値にする
* mixed_layer - 混合層深の計算

### 渦度, 地衡流

* ertel_pv    - Ertel PV
* iptw_pvz    - Ertel PV を層厚重みつき平均により等密度面上の値として求める
* geostrophic - 地衡流の計算

### その他の解析

* spectrum    - スペクトル解析
* tracer_offline - オフラインでトレーサー流し実験を行う

### 潮汐

* naotide     - Matsumoto et al. (2000)の潮汐同化データ
* tide        - 潮汐の解析

### 衛星

* modis	      - MODISデータ

### 作業支援

* utils       - ファイル操作のためのシェル・スクリプト
* file_mng    - ファイルIOのサポートルーチン
* test_lib    - MXEライブラリのテスト
