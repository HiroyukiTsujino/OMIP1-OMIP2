World Ocean Atlas 2013 version 2
========

変数
--------

  * Temperature (t)
  * Salinity (s)
  * Dissolved Oxygen (o)
  * Percent Oxygen Saturation (O)
  * Apparent Oxygen Utilization (A)
  * Silicate (i)
  * Phosphate (p)
  * Nitrate (n)


時間解像度
--------

  * annual (00)
  * monthly (1-12月 = 01-12)
  * seasonal 
    + Winter(1-3月) = 13
    + Spring(4-6月) = 14
    + Summer(7-8月) = 15
    + Autumn(9-12月)= 16


水平解像度
--------

  * 5度 => 未取得
  * 1度 (全変数あり)
  * 1/4度 (温度、塩分のみ)


鉛直解像度
--------

  * 102層

  * annual = 102層 (0-5500m)
  * seasonal 
    + Silicate, Phosphate, Nitrate: 37層 (0-500m)
    + その他                       : 102層 (0-5500m)
  * monthly
    + Silicate, Phosphate, Nitrate: 37層 (0-500m)
    + その他                       : 57層 (0-1500m)


その他
--------

  * 温度、塩分は全期間平均値(decav)に加え, 6つの10年平均値も整備
    - decav = Averaged decades
    - 5564 = 1955-1964 years
    - 6574 = 1965-1974 years
    - 7584 = 1975-1984 years
    - 8594 = 1985-1994 years
    - 95A4 = 1995-2004 years
    - A5B2 = 2005-2012 years
  * その他変数は全期間平均値(decav)のみを整備


オリジナルデータ
--------

  /denkei-shared/og/ocpublic/refdata/WOA/WOA13v2_surakawa/original

  * 1deg => 1度格子 (NetCDF形式)
  * 025deg => 1/4度格子 (NetCDF形式)
  * masks => mask データ (CSV形式)

grads処理
--------

  ogsv007:~surakawa/GONDOLA_100/prep/gondola_100/woa13v2_grads


GrADS データ
--------

  陸地埋めなし /denkei-shared/og/ocpublic/refdata/WOA/WOA13v2_surakawa/grads
  陸地埋めあり /denkei-shared/og/ocpublic/refdata/WOA/WOA13v2_surakawa/grads_filled

  * monthly データ
    - 深層部は annual (Silicate, Phosphate, Nitrate) or seasonal (その他) を使用
    - woa13_[dec]_t.ctl     : Temperature
    - woa13_[dec]_s.ctl     : Salinity
    - woa13_[dec]_th.ctl    : Potential Temperature
    - woa13_[dec]_rho.ctl   : in-situ Density Anomaly [kg/m3 - 1000]
    - woa13_[dec]_sigma0.ctl: Potential Density Anomaly with reference pressure of    0 dbar
    - woa13_[dec]_sigma2.ctl: Potential Density Anomaly with reference pressure of 2000 dbar
    - woa13_[dec]_sigma4.ctl: Potential Density Anomaly with reference pressure of 4000 dbar
    - woa13_all_o.ctl       : Dissolved Oxygen
    - woa13_all_O.ctl       : Percent Oxygen Satulation
    - woa13_all_A.ctl       : Apparent Oxygen Utilization
    - woa13_all_i.ctl       : Silicate
    - woa13_all_p.ctl       : Phosphate
    - woa13_all_n.ctl       : Nitrate
    - [dec] = decav, 5564, 6574, 7584, 8594, 95A4 or A5B2

  * annual データ
    - woa13_*_*_ann.ctl

  * seasonal データ
    - Silicate, Phosphate, Nitrate 深層部は annual データを使用
    - woa13_*_*_ssn.ctl

  * basinmask データ (/denkei-shared/og/ocpublic/refdata/WOA/WOA13v2_surakawa/grads/masks)
    - basinmask_01.ctl : 1度格子用
    - basinmask_04.ctl : 1/4度格子用

    - Basin index list (括弧内は海盆が定義される最も浅い標準深度レベル)
      +  1: Atlantic Ocean             (1*)
      +  2: Pacific Ocean              (1*)
      +  3: Indian Ocean               (1*)
      +  4: Mediterranean Sea          (1*)
      +  5: Baltic Sea                 (1)
      +  6: Black Sea                  (1)
      +  7: Red Sea                    (1)
      +  8: Persian Gulf               (1)
      +  9: Hudson Bay                 (1)
      + 10: Southern Ocean             (1*)
      + 11: Arctic Ocean               (1)
      + 12: Sea of Japan               (1)
      + 13: Kara Sea                   (22)
      + 14: Sulu Sea                   (25)
      + 15: Baffin Bay                 (37)
      + 16: East Mediterranean         (41)
      + 17: West Mediterranean         (47)
      + 18: Sea of Okhotsk             (47)
      + 19: Banda Sea                  (55)
      + 20: Caribbean Sea              (55)
      + 21: Andaman Basin              (62)
      + 22: North Caribbean            (67)
      + 23: Gulf of Mexico             (67)
      + 24: Beaufort Sea               (77)
      + 25: South China Sea            (77)
      + 26: Barents Sea                (77)
      + 27: Celebes Sea                (62)
      + 28: Aleutian Basin             (77)
      + 29: Fiji Basin                 (82)
      + 30: North American Basin       (82)
      + 31: West European Basin        (82)
      + 32: Southeast Indian Basin     (82)
      + 33: Coral Sea                  (82)
      + 34: East Indian Basin          (82)
      + 35: Central Indian Basin       (82)
      + 36: Southwest Atlantic Basin   (82)
      + 37: Southeast Atlantic Basin   (82)
      + 38: Southeast Pacific Basin    (82)
      + 39: Guatemala Basin            (82)
      + 40: East Caroline Basin        (87)
      + 41: Marianas Basin             (87)
      + 42: Philippine Sea             (87)
      + 43: Arabian Sea                (87)
      + 44: Chile Basin                (87)
      + 45: Somali Basin               (87)
      + 46: Mascarene Basin            (87)
      + 47: Crozet Basin               (87)
      + 48: Guinea Basin               (87)
      + 49: Brazil Basin               (92)
      + 50: Argentine Basin            (92)
      + 51: Tasman Sea                 (87)
      + 52: Atlantic Indian Basin      (92)
      + 53: Caspian Sea                (1)
      + 54: Sulu Sea II                (37)
      + 55: Venezuela Basin            (37)
      + 56: Bay of Bengal              (1*)
      + 57: Java Sea                   (16)
      + 58: East Indian Atlantic Basin (97)

履歴
--------

  * 2016/06/30 オリジナルデータ取得 (averaged decades/all, 1deg/025deg)
  * 2016/10/20 GrADS形式変換、陸地埋め
  * 2016/10/28 annual/seasonal データの GrADS形式変換、陸地埋め
  * 2017/12/27 密度変数を追加 
               MXE prep/refdata/WOA で処理
               リビジョン：d435d17e04088bf41eff37c52aec05692a26f225
  * 2018/08/30 各decade毎のT/Sデータを追加
  * 2019/05/22 basinmaskデータ取得、GrADS形式変換
