
* Trial Number - iteration counter for repeating the same set of hyperparameters

| optim | G lr | D lr | Batch size (+offset) | Seq len | Train Eps | G Pre-train Eps | D Pre-train Eps | G loss | D loss | Result | Trial Number | Output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SGD | 0.01 | 0.005 | 10(0) | 256 | 300 | 5 | 5 | Feature matching | Simple | Fail | 1 | Not saved |
| SGD | 0.01 | 0.005 | 32(16) | 256 | 300 | 5 | 5 | Feature matching | Simple | Fail | 1 | Not saved |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | Simple | Relatively STABLE training, mode collapse | 1 | Not saved |
| SGD | 0.001 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | Simple | Unstable training, output quite repetitive and notes are too high | 1 | Not saved |
| SGD | 0.005 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | Simple | A bit unstable training, output not playing (probably because of high start tick) | 1 | Not saved |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | w/ label smoothing | Relatively STABLE training, sounds minor | 1 | label_smoothing.zip |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 20 | Feature matching | w/ label smoothing | Relatively STABLE training, sounds minor, fast tick, monotonous | 1 | Not saved |
| SGD | 0.01 | 0.01 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | w/ label smoothing | A bit unstable training, high start tick | 1 | Not saved |
| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Simple | w/ label smoothing | Unstable | 1 | Not saved |

| Adam | 0.001 | 0.001 | 16(2) | 256 | 300 | 5 | 5 | Simple | w/ label smoothing | Unstable | 1 | Not saved |
| Adam | 0.001 | 0.001 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | w/ label smoothing | Unstable | 1 | Not saved |
| SGD | 0.0001 | 0.0004 | 16(2) | 256 | 300 | 10 | 10 | Simple | w/ label smoothing | STABLE, but it seems no learning | 1 | Not saved |
| SGD | 0.001 | 0.001 | 16(2) | 256 | 300 | 10 | 10 | Simple | w/ label smoothing | STABLE, starts out fine, but mode collapse in latter time steps | 1 | sgd_low_lr.zip |
| SGD | 0.001 | 0.001 | 16(2) | 256 | 500 | 10 | 10 | Simple | w/ label smoothing | STABLE, mode collapse | 1 | Not saved |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 300 | 10 | 10 | Simple | w/ label smoothing | A bit unstable training, sounds like going up & down on the scale | 1 | adam_low_lr.zip |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 500 | 10 | 10 | Simple | w/ label smoothing | Minimally unstable training, sounds like banging on the piano | 1 | Not saved |

| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 300 | 10 | 10 | Feature matching | w/ label smoothing | A bit unstable training, output quite repetitive | 1 | adam_low_lr_feat.zip |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 400 | 10 | 10 | Feature matching | w/ label smoothing | Unstable training, high start tick | 1 | Not saved |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 300 | 10 | 10 | Simple | w/ label smoothing | Unstable training, short output | 2 | Not saved |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 400 | 10 | 10 | Simple | w/ label smoothing | Unstable training, repetitive output | 1 | Not saved |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 300 | 10 | 10 | Simple | w/ label smoothing | Unstable training, monotonous | 3 | Not saved |
| Adam | 0.0001 | 0.0004 | 16(2) | 256 | 200 | 10 | 10 | Simple | w/ label smoothing | A bit unstable training, sounds minor | 1 | |

| SGD | 0.01 | 0.005 | 16(2) | 256 | 300 | 5 | 5 | Feature matching | w/ label smoothing | A bit unstable, high start tick | 2 | Not saved |
| Adam | 0.001 | 0.001 | 10(0) | 256 | 300 | 5 | 5 | Simple | w/ label smoothing | unstable, collapses eventually | 1 | Not saved |

