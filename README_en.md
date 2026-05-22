# Graph-autofusion

## 🔥Latest News

- [2026/04] Autofuse component is open sourced!
  The Autofuse automatic fusion function is provided on Ascend chips. It can automatically fuse adjacent operators into one, eliminating the time consumption of input and output transfer, reducing the number of operators, and optimizing the total operator execution time.
- [2025/10] Graph-autofusion project is open sourced!
  SuperKernel fusion function is provided on Ascend chips. It can reduce task scheduling waiting time and scheduling overhead, optimizing operator execution overhead.

## 🚀Overview

Graph-autofusion is a lightweight, decoupled component collection for Ascend chips. It aims to accelerate model execution through various fusion-related technologies. Currently, the SuperKernel component and Autofuse component are open sourced, and more fusion-related modules will be continuously released in the future.

Component features:

- **Focus on fusion acceleration technology**: Efficient fusion and acceleration are implemented based on codegen JIT compilation mechanism.
- **Modular and decoupled**: Components are independent and can be selected as needed. The underlying dependencies are minimal, relying only on AscendC and the runtime environment.

## ⚡️Quick Start

- To experience the complete build, test, and sample running process of Graph-autofusion, see [Build Verification](docs/en/build.md).
- To understand the principle and usage of the SuperKernel component, see [SuperKernel Introduction](super_kernel/README.md).
- To understand the principle and usage of the Autofuse component, see [Autofuse Introduction and Quick Start](autofuse/README.md).

## 🔍Directory Structure

```
├── autofuse                       # Autofuse component. Autofuse source code, tests, and documentation are in this subdirectory.
├── build.sh                       # One-click project build script
├── cmake                          # Project build directory
├── CMakeLists.txt                 # Project CMakeLists
├── docs                           # Project overall documentation
│  ├── build.md                    # One-click build script documentation
│  ├── ...                         # Other documentation related to this project
├── super_kernel                   # SuperKernel component. SuperKernel source code, tests, and documentation are in this subdirectory.
├── ...                            # Planned future components
├── README.md                      # graph-autofusion project overall functionality introduction
├── scripts                        # Script path
   └── package
```

## 📝Related Information

- [Contributing Guide](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [License](LICENSE)