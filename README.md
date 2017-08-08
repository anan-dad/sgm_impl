# SGM_impl

This is an implementation of Semi Global Matching (SGM) using the message passing algrithm which corresponds to my solution of a lab course in University of Freiburg. Result disparity maps are included.

## Key idea

- Inward pass (by dynamic programming)
- Outward pass (again by dynamic programming)
- Combination of messages

## Benchmark


tsukuba | couch | venus | 1_img | 2_img | 3_img | 4_img | 5_img | 6_img | 7_img
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
5.531s | 15.626s | 15.679s | 24.121s | 25.296s | 24.203s | 26.486s | 22.608s | 28.027s | 23.222s

## Disparity maps


tsukuba | couch
-- | -- | -- 
<img src="imgs/tsukuba.png" width="500">  | <img src="imgs/couch.png" width="500">

venus | 1-img
-- | -- | -- 
<img src="imgs/venus.png" width="500">  | <img src="imgs/1-img.png" width="500">

2-img | 3-img
-- | -- | -- 
<img src="imgs/2-img.png" width="500">  | <img src="imgs/3-img.png" width="500">

4-img | 5-img
-- | -- | -- 
<img src="imgs/4-img.png" width="500">  | <img src="imgs/5-img.png" width="500">

6-img | 7-img
-- | -- | -- 
<img src="imgs/6-img.png" width="500">  | <img src="imgs/7-img.png" width="500">

## References

- [Drory, Amnon, et al. "Semi-global matching: a principled derivation in terms of message passing." German Conference on Pattern Recognition. Springer, Cham, 2014.](https://link.springer.com/chapter/10.1007/978-3-319-11752-2_4)
- [Hirschmüller, Heiko. "Semi-global matching-motivation, developments and applications." Photogrammetric Week 11 (2011): 173-184.](http://elib.dlr.de/73119/)
