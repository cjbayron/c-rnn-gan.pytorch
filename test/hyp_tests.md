

| optim | G lr | D lr | Batch size (+offset) | Seq len | Train Eps | G Pre-train Eps | D Pre-train Eps | G loss | Result | Tries | Output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SGD | 0.01 | 0.005 | 10(0) | 256 | 300 | 5 | 5 | Feature matching | Fail | 1 | Not saved |
| SGD | 0.01 | 0.005 | 32(16) | 256 | 300 | 5 | 5 | Feature matching | Fail | 1 | Not saved |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | Relatively STABLE training, mode collapse | 1 | Not saved |
| SGD | 0.001 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | Unstable training, output quite repetitive and notes are too high | 1 | Not saved |
| SGD | 0.005 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | A bit unstable training, output not playing (probably because of high start tick) | 1 | Not saved |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching (w/ label smoothing) | Relatively STABLE training, sounds minor | 1 | label_smoothing.zip |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 20 | Feature matching (w/ label smoothing) | Relatively STABLE training, sounds minor, fast tick, monotonous | 1 | Not saved |
| SGD | 0.01 | 0.01 | 16(2) | 256 | 300 | 5 | 5 | Feature matching (w/ label smoothing) | | | |
