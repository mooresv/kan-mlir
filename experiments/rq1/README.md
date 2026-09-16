# RQ1: Post-Training Representation Flexibility

This directory contains experiments investigating the first research
question of the KAN compiler project:


The central hypothesis is that the function learned during training
does not need to remain tied to the computational representation used
to parameterize that function during training.

For KANs, the learned B-spline edge functions provide a convenient
case study. After training, each learned spline can be transformed into
an alternative piecewise-polynomial representation without retraining
the network.

The experiments in this directory evaluate both:

1. an effectively exact knot-aligned piecewise-cubic transformation,
   and
2. approximate piecewise-cubic representations using fewer pieces.

The goal is to characterize the tradeoff between representation
complexity, numerical fidelity, and task-level accuracy.


## Experiment 1: Nonlinear Regression

The first experiment trains a three-layer KAN on the synthetic
regression problem

$$y = \sin(\pi x_1 x_2) + (x_3 - 0.5)^2 + 0.5x_4,$$

where each input is sampled uniformly from

$$
x_i \in [-1,1].
$$

The network architecture is

```text
4 -> 16 -> 16 -> 1
```

using cubic B-spline KANLib Linear layers with grid size 5.

The model contains 336 leanred spline edge functions:

```text
4 x 16 + 16 x 16 + 16 x 1 = 336.
```

### Data Separation

The experiment uses three independent datasets:

|  |  |
|---|---|
| Training: |    10,000 examples |
| Calibration: |  2,500 examples |
| Test:        |  5,000 examples |

The training set is used only for model optimization.

The calibration set is used after training to determine the observed
activation range of each input feature of each layer. These ranges are
used to construct the approximate polynomial representations.

The test set is not used for training, calibration-range selection, or
polynomial fitting. It is used only for final evaluation.

### Training

Run:

```bash
cd experiments/rq1
python train_kan_regression.py
```

The script trains the KAN for 100 epochs using minibatch adam and saves

```bash
trained_kan_regression.pt
```

containing the trained model and the training, calibration, and test datasets.

### Post-Training Representation Transformation

Run:

```bash
python transform_trained_regression.py
```

No retraining is performed during or after transformation.

The program constructs five alternative representations of every learned
spline edge:

```text
1-piece cubic approximation
2-piece cubic approximation
4-piece cubic approximation
8-piece cubic approximation
exact knot-aligned piecewise cubic
```

For approximate representations, each input feature's observed
calibration range is divided uniformly into the requested number of
pieces. A cubic polynomial is fitted independently on each piece using
least squares.

The exact representation uses every interval in the extended B-spline
knot vector. With grid size 5 and spline order 3, the KANLib knot
vector contains 12 knots and therefore 11 polynomial intervals.

The residual SiLU branch of each KANLib layer is preserved unchanged.
Only the learned spline contribution is transformed.

### Exact B-Spline Support

For the configuration used here, KANLib constructs the extended knot
vector

```text
[-2.2, -1.8, -1.4, -1.0, -0.6, -0.2,
  0.2,  0.6,  1.0,  1.4,  1.8,  2.2]
```

Although the nominal spline range is $[-1,1]$, learned spline
functions can be evaluated over the complete extended support.

Consequently, the exact transformation uses all 11 knot intervals
rather than only the five intervals in the nominal spline range.


### Current Regression Result

The trained original KAN achieves

```text
Test MSE  = 1.407415693393e-04
Test RMSE = 1.186345517635e-02
```

The post-training representations produce:

| Representation | Pieces | Task RMSE | RMS Difference from Original | Relative RMS Difference |
|---|---:|---:|---:|---:|
| Original B-spline | -- | 1.1863455e-02 | -- | -- |
| 1-piece cubic | 1 | 1.6671249e-01 | 1.6858245e-01 | 1.5074526e-01 |
| 2-piece cubic | 2 | 1.9319691e-02 | 1.6912464e-02 | 1.5123007e-02 |
| 4-piece cubic | 4 | 1.1921985e-02 | 1.4011690e-03 | 1.2529155e-03 |
| 8-piece cubic | 8 | 1.1868669e-02 | 1.9116864e-04 | 1.7094167e-04 |
| Exact knot-aligned | 11 | 1.1863488e-02 | 1.9071082e-07 | 1.7053229e-07 |

The exact knot-aligned transformation therefore reproduces the
complete three-layer trained network to approximately FP32 numerical
precision.

The 4-piece representation increases task RMSE by approximately
0.49%, while the 8-piece representation increases task RMSE by
approximately 0.044%, without retraining.


### Error Propagation

The program reports two types of error.

### Local Layer Representation Error

For layer $l$, both the original and transformed layer receive the
same original activation:

$$
\tilde{F}_l(x_l) - F_l(x_l).
$$

This isolates the error introduced by the representation change in
that layer.

### Accumulated Network Error

The transformed network consumes its own transformed activations:

$$\tilde{x}_{l+1} = \tilde{F}_l(\tilde{x}_l).$$

This measures how representation errors propagate through the complete
network.

For the exact knot-aligned representation, the final accumulated RMS
difference is

```text
1.907108214e-07
```

after transforming all 336 spline edge functions.


### Calibration-Range Coverage

The approximate representations use only calibration-set activation
ranges.

On the independent test set, the fractions of activations outside
those ranges were:

| Layer input | Outside calibration range |
|---|---:|
| Layer 0 | 0.0700% |
| Layer 1 | 0.07375% |
| Layer 2 | 0.0800% |

The maximum excursions outside the calibration ranges were
approximately `0.0018`, `0.0835`, and `0.1030`, respectively.

For approximate representations, values outside the calibration range
are evaluated by extrapolating the nearest end polynomial piece.


### Interpretation

These results demonstrate two forms of post-training representation
flexibility.

First, a trained cubic B-spline KAN can be converted to a
knot-aligned piecewise-polynomial representation with only
floating-point-level numerical differences.

Second, the learned functions can be approximated using substantially
fewer polynomial regions while preserving task-level accuracy. The
4-piece and 8-piece representations retain nearly the same regression
accuracy as the original trained model.

The number of polynomial pieces should not be interpreted as model
parameter compression. A cubic polynomial requires four coefficients
per piece. Instead, the experiment establishes a family of
computational representations with different evaluation structures and
different approximation errors.

Whether these representations provide different execution costs on
specific hardware is investigated separately under RQ2.


### Files

```bash
train_kan_regression.py
```

Trains the regression KAN and generates the training, calibration, and
test datasets.

```bash
transform_trained_regression.py
```

Constructs the 1-, 2-, 4-, 8-piece and exact knot-aligned polynomial
representations and evaluates their numerical and task-level errors.

```bash
check_regression_ranges.py
```

Diagnostic program used to inspect the activation ranges of the
trained multilayer KAN.

### Conclusions

The regression experiments has demonstrated the following:

> **A trained multilayer KAN can be transformed post-training, without retraining, from its learned B-spline parameterization into alternative piecewise polynomial representations. An exact knot-aligned transformation reproduces the original model to FP32 precision, while approximate representations provide a controllable complexity-fidelity tradeoff; in this experiment, four and eight pieces preserve task RMSE within 0.5% and 0.05%, respectively.

## Planned RQ1 Experiments

The next RQ1 experiment will evaluate the same post-training
transformation methodology on a classification task. This will allow
measurement of both classification accuracy and prediction agreement
between the original and transformed networks.

