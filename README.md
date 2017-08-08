# SGM_impl

This is an implementation of Semi Global Matching (SGM) using the message passing algrithm which corresponds to my solution of a lab course GPU programming in university of Freiburg.

## Key idea

- Inward pass (by dynamic programming)
- Outward pass (again by dynamic programming)
- Combination of messages

## Benchmark


tsukuba | couch | venus | 1_img | 2_img | 3_img | 4_img | 5_img | 6_img | 7_img

5.531 sec | 15.626 sec | 15.679 sec | 24.121 sec | 25.296 sec | 24.203 sec | 26.486 sec | 22.608 sec | 28.027 sec | 23.222 sec

## References

- [Drory, Amnon, et al. "Semi-global matching: a principled derivation in terms of message passing." German Conference on Pattern Recognition. Springer, Cham, 2014.](https://link.springer.com/chapter/10.1007/978-3-319-11752-2_4)
- [Hirschmüller, Heiko. "Semi-global matching-motivation, developments and applications." Photogrammetric Week 11 (2011): 173-184.](http://elib.dlr.de/73119/)
