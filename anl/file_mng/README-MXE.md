file_mng
========

ファイル IO サポートルーチン


ドキュメント
--------

  * fileunmng.F90 : Fortran 装置番号の管理(121-199)
  * fileocmng.F90 : ファイルのopen/closeを行う。
  * 使用例
      use file_open_close_manager
      integer(4) :: mtin1
      call open_file_direct(mtin1, flnin1, 4*imut*jmut)
      call close_file(mtin1)

開発
-------

  * 開発：気象研究所 海洋・地球化学研究部 第一研究室
  * 窓口：辻野 htsujino@mri-jma.go.jp
