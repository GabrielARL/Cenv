# Ccode.jl 📡  
*A minimal Julia toolkit for convolutional coding, ISI channels, and hybrid decoding (BCJR + DFE).*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-v1.10%2B-green.svg)](https://julialang.org)

---

## 🧩 Overview

**Ccode.jl** provides a compact and readable simulation framework for convolutional coding and decoding over intersymbol interference (ISI) channels.  
It’s designed for research, prototyping, and teaching — especially in underwater acoustic or multipath environments where **joint equalization and decoding** is critical.

---

## 🚀 Features

| Capability | Description |
|-------------|-------------|
| **Convolutional Encoding** | Generate systematic RSC trellises, e.g., (5,7) octal, K=3 |
| **BCJR Equalization** | Optimal MAP decoding over ISI channels (`decode_info_over_isi_lti`) |
| **MMSE DFE** | Fast linear Decision-Feedback Equalizer front-end (`decode_info_over_isi_lti_dfe`) |
| **Noise-Preserving Swap** | Transform noisy signals between codewords without altering the noise |
| **Utility Functions** | `fullconv`, `sameconv`, `hardbits`, `jac`, `clampllr`, and more |
| **Demos & Tests** | End-to-end examples and a lightweight test suite |

---

## 📂 Repository Layout

