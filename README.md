# lecture_polymphys_langevin

講義「高分子物理学特論」の#7「Dynamics of a Single Chain」で利用

研究室内モノグラフでも利用できそう、v2で実装したい

自己相関関数の計算は以下を参考に作成
https://tech.gijukatsu.com/numpy_autocorrelation/

Langevin方程式を差分で解くためにオイラー・丸山法に関する以下の記事参考にした。
https://qiita.com/chemweb000/items/1a7333bc485fb36cfb5f

＜未解決問題＞ 
突貫工事だったので精査しきれていない。そもそも的にum、msを単位として計算をしている風だけど、今のパラメータで完全にあっているかどうかチェックが必要
自己相関関数は一つのアレイをずらしていって計算するというやり方なので、後半のデータは（このやり方なら本質的に）信頼性が落ちる、この点は修正すべき
