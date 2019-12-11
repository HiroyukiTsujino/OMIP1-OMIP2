history データの空間積分や時間平均を行う (anal/integ)
========


内容
--------

使い方はサンプル(sample/)のシェル・スクリプトを参照。

  * 2つの実験の海面高度差 (make_ssh_diff.sh)
  * 2つの実験の海面高度差(時間平均) (make_ssh_diff_mean.sh)
  * 海面高度の領域平均 (make_ssh_have.sh)
  * 海面高度の時間平均 (make_ssh_mean.sh)

  * MRI.COM モニター用のマスクをかけたうえで水平平均を行う (make_ssh_have-mask.sh)

  * 30分毎データを日平均に (make_30min2day.sh)
  * 日毎データを月平均に (make_day2month_1grid.sh)
  * 5日毎データを月平均に (make_5day2month.sh)

  * 日別値から日毎平年値を作成する (make_climatology.sh)

  * 鉛直積分 (make_integ_vert.sh)
  * 運動エネルギー全領域積分の時系列 (ke_vol.sh)
  * 順圧運動エネルギーの体積積分 (make_barotropic_ke_vol.sh)


Fortran program
--------

  * 2つのデータの差 (diff_ctl.F90)
  * 2つのデータの差を時間平均 (diff_mean_ctl.F90)
    operate="square" とすると差を2乗して平均する
  * 水平平均 (have_ctl.F90)
    オプションでマスク配列を指定できる。
    マスク配列を作るサンプル・プログラム: sample/mask.F90
    オプションoperateを"square"とすると2乗して水平平均する
  * 時間平均 (mean_ctl.F90)
    operate="square" とすると値を2乗して平均する
  * 時間移動平均 (runmean_ctl.F90)
  * 体積平均 (vave_ctl.F90)
