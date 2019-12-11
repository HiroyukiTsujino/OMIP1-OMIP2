
# OMIP1-OMIP2 比較論文用 データ補間・形式変換ツール
---

## ファイル

* link_make_dir.sh: リンク・ディレクトリ作成
* copy_log.sh     : データコピー

* intp_hm.py   : 水平平均温位・塩分 鉛直補間
* intp_zm.py   : 東西平均温位・塩分 南北・鉛直補間
* intp_nht.py  : 南北熱輸送       南北補間
* intp_moc.py  : z座標MOC        南北・鉛直補間
* intp_mocip.py: 密度座標MOC      南北補間

* convert_amoc_rapid.py: 26.5N AMOC 最大流量       補間・形式変換
* convert_mfo.py       : ドレーク海峡通過流量、ITF流量 形式変換
* convert_scalar.py    : スカラー量データ            形式変換
* convert_2d.py        : 水平2次元データ             形式変換


* scalar.json: スカラー量データ用 json ファイル
* 2d.json    : 水平2次元データ用 json ファイル


## 担当

全球大気海洋研究部 第四研究室 浦川 surakawa@mri-jma.go.jp


## 作業履歴

2019/08/09 変換
