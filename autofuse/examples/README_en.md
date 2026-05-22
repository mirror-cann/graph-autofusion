# Autofuse Sample Usage Guide

## Function Description

Use autofuse to complete various types of operator fusion.

## Directory Structure
```
├── af_pointwise                  # pointwise type operator fusion sample
│  └── af_add_ge.py               # complete fusion of add and ge two pointwise type operators through autofuse
├── af_reduce                     # reduce type operator fusion sample
│  └── af_mul_reducesum.py        # complete fusion of mul and reducesum two pointwise type operators through autofuse
```

## Prerequisites

Please refer to "[Autofuse Introduction and Quick Start](../README.md)" to complete prerequisite environment preparation.

## Use Case Demonstrations

[Use Case 1](af_pointwise/README.md)

[Use Case 2](af_reduce/README.md)

## Reference

Please refer to relevant content in [inductor-npu-ext User Manual](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/docs/manuals.md).