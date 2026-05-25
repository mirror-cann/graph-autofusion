# Build from Source

## 1. Environment Preparation

Complete the basic environment setup before compiling from source. See [Quick Install](quick_install.md) for detailed instructions.

## 2. Environment Verification

Verify the environment after installing the CANN packages.

```bash
# Check CANN Toolkit version (default installation path). <arch> represents CPU architecture (aarch64 or x86_64). For WebIDE scenarios, replace /usr/local with /home/developer.
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# Check CANN ops version (default installation path). <opsname> represents the ops subpackage name to query. For WebIDE scenarios, replace /usr/local with /home/developer.
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## 3. Environment Variable Configuration

Select the appropriate command based on your scenario:

  ```bash
  # Default installation path (root user example). For non-root users, replace /usr/local with ${HOME}.
  source /usr/local/Ascend/cann/set_env.sh
  # Custom installation path
  source ${install_path}/cann/set_env.sh
  ```

## 4. Source Code Compilation

### 4.1 Download Source Code

If your compilation environment can access the network, open-source third-party software downloads automatically during compilation. Use the following command to download the source code:

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
 ```

If your compilation environment cannot access the network, download the source code and open-source software packages in a networked environment. Upload them manually to your compilation environment:

1. In a networked environment, visit the [project homepage](https://gitcode.com/cann/graph-autofusion). Use the `Download zip` or `Clone` button to download the source code.

2. Download the `makeself` and `cann-cmake` third-party open-source software.

   | Open-Source Software | Version | Download Link |
   |---|---|---|
   | makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
   | cann-cmake | master-002 | [cmake-master-002.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-002.tar.gz) |

3. Upload the source code and third-party open-source software to the offline compilation environment. Extract the source code.

   ```bash
   # Extract source code
   unzip graph-autofusion.zip
   ```

4. Create an open_source directory at the same level as the source code. Place the third-party open-source software packages in the open_source directory and extract them.

   ```bash
   # Create open_source directory
   mkdir open_source
   # Move third-party open-source software packages to open_source directory
   mv makeself-release-2.5.0-patch1.tar.gz open_source/
   mv cmake-master-002.tar.gz open_source/
   # Extract third-party open-source software packages and rename (cann-cmake package remains unchanged)
   cd open_source
   tar -zxvf makeself-release-2.5.0-patch1.tar.gz && mv makeself-release-2.5.0 makeself
   ```

5. The file directory structure after completion:

   ```bash
   ├── graph-autofusion           # Extracted source repository
   │  ├── cmake
   │  └── ...
   ├── open_source                # Third-party open-source software
   │  └── makeself
   │     ├── COPYING
   │     ├── Makefile
   │     ├── README.md
   │     └── ...
   │  └── cmake-master-002.tar.gz
   ```

### 4.2 Install Dependencies

#### 4.2.1 Install Dependencies

The following lists dependencies used for source code compilation. Pay attention to version requirements:
> [!NOTE] Note
> If you use the mirror method for project experience, all dependencies are included in [init_env.sh](../../scripts/init_env.sh). You can skip this dependency installation step.

- Python3 >= 3.8.0 (Python virtual environment recommended)
  > [!NOTE] Note
  > - Python announced 3.8.x EOL. CANN will stop support for this version in 10.0.0. Please upgrade to >= 3.9.x.

  1. Create and activate a virtual environment:

     ```shell
     python3 -m venv venv
     source venv/bin/activate
     ```

  2. Install dependencies:

     ```shell
     cd graph-autofusion
     pip3 install -r super_kernel/requirements-dev.txt
     ```

- CMake >= 3.16.0  (version 3.20.0 recommended)

   ```shell
   # Ubuntu/Debian installation example. For other operating systems, install manually.
   sudo apt-get install cmake
   ```

#### 4.2.2 Check Compilation Environment

After environment preparation, execute the environment check script. Confirm whether the current environment meets compilation requirements.

```bash
bash scripts/check_env.sh
```

Check result descriptions:

| Status | Meaning | Recommendation |
|---|---|---|
| **[PASS]** | Check passed | No action required |
| **[WARNING]** | Non-critical dependency missing or version deviation | Fix recommended, does not affect core compilation |
| **[ERROR]** | Critical dependency missing or version incompatible | Must fix, otherwise compilation cannot proceed |

> [!NOTE] Note
> All check items and version constraints in the environment check script come strictly from docs/build.md and super_kernel/requirements-dev.txt. If documentation and dependencies update, synchronize changes to the [script](../../scripts/check_env.sh).

### 4.3 Compilation

Navigate to the source repository root directory. Execute the following command to compile:

```shell
bash build.sh --pkg
```

View more compilation parameters through `bash build.sh -h`. After successful execution, `cann-graph-autofusion_${cann_version}_linux-${arch}.run` generates in the `build_out` directory.

- --pkg indicates building a run package.
- ${cann_version} indicates the cann version number.
- ${arch} indicates CPU architecture, for example, aarch64 or x86_64.

During compilation, CMake automatically downloads third-party source packages (abseil-cpp, boost, json, protobuf, symengine, and so on) required by autofuse from external networks (`https://gitcode.com/cann-src-third-party/`) through `ExternalProject_Add`. If your compilation environment cannot directly access external networks (for example, enterprise intranet or Docker container default bridge network), select the appropriate solution:
 
**Solution 1: Configure Network Proxy (Recommended)**
If the environment has an HTTP proxy that can access external networks, set the `http_proxy` / `https_proxy` environment variables before compilation. `build.sh` automatically passes these proxy variables to CMake subprocesses:

```shell
# Set proxy (modify according to actual proxy address)
export http_proxy=http://user:password@proxy-server:port
export https_proxy=http://user:password@proxy-server:port

# Execute compilation
bash build.sh --pkg
```

> [!NOTE] Note
> - `build.sh` automatically detects and inherits `http_proxy` and `https_proxy` environment variables in the current shell. No additional configuration required.
> - If you use git proxy (configured through `git config --global http.proxy`), confirm that the shell environment also sets corresponding environment variables. Verify through `echo $http_proxy`.

**Solution 2: Manually Pre-download Third-Party Packages**
In environments without any external network access, pre-download third-party source packages on a networked machine. Copy them to the specified directory in the compilation environment for offline compilation.

1. Download the following third-party packages on a networked machine:

   | Third-Party Package | Version | Download Link |
   |---------|------|---------|
   | abseil-cpp | 20230802.1 | https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz |
   | json | 3.11.3 | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz |
   | boost | 1.87.0 | https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz |
   | protobuf | 25.1 | https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz |
   | symengine | 0.12.0 | https://gitcode.com/cann-src-third-party/symengine/releases/download/v0.12.0/symengine-0.12.0.tar.gz |

2. Copy the downloaded packages to the corresponding subdirectories under `output/third_party/` in the compilation environment (create if not exist):

   ```shell
   # Create directory structure under source root
   mkdir -p output/third_party/{abseil-cpp,json,boost,protoc,symengine}

   # Place downloaded packages in corresponding directories (filenames must match the table below)
   # abseil-cpp-20230802.1.tar.gz  → output/third_party/abseil-cpp/
   # json-3.11.3.tar.gz            → output/third_party/json/
   # boost_1_87_0.tar.gz           → output/third_party/boost/
   # protobuf-25.1.tar.gz          → output/third_party/protoc/
   # symengine-0.12.0.tar.gz       → output/third_party/symengine/
   ```

3. During compilation, specify the local path through `--cann_3rd_lib_path` to skip the download step:

   ```shell
   bash build.sh --pkg --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

> [!NOTE] Note
> - If you do not specify `--cann_3rd_lib_path`, the default search path is `./output/third_party`. Therefore, you can omit this parameter when packages exist in the default path.
> - CMake build scripts prioritize checking whether corresponding tarballs already exist in the local path. If they exist, the download skips.

### 4.4 Test Verification

After compilation, you can perform developer testing. Ensure you have completed [Environment Preparation](./quick_install.md#1-environment-preparation) before executing operations in this section.

- UT Verification

   ```bash
   bash build.sh -u
   ```

   After execution, check the UT test execution status through the output log. Successful test case execution prints `passed` without any `failed` print. Confirm all test cases pass.

- ST Verification

   ```bash
   bash build.sh -s
   ```

   After execution, check the ST test execution status through the output log. Successful test case execution prints `passed` without any `failed` print. Confirm all test cases pass.

- Coverage Verification

   ```bash
   bash build.sh -u -c  # UT coverage
   bash build.sh -s -c  # ST coverage
   bash build.sh -c     # UT coverage + ST coverage
   ```

   After execution, check the coverage status through the output log. Confirm all test cases pass.

### 4.5 Installation and Uninstallation

- Installation

  After local verification completes, execute the following command to install the compiled package. Ensure the installation user has execute permission for the package.

  ```shell
  # Specify installation path if needed: --install-path=${install_path}
  ./cann-graph-autofusion_${cann_version}_linux-${arch}.run --full --quiet --pylocal
  ```

  > [!NOTE] Note
  > - The installation path (default or specified) must match the path where you installed the cann-toolkit package.
  > - --full indicates full installation mode.
  > - --install-path specifies the installation path. If not specified, the default installation path is `/usr/local/Ascend` (root user) or `${HOME}/Ascend` (non-root user).
  > - --quiet indicates silent installation. It skips human-computer interaction.  
  > - --pylocal determines whether to install .whl files inside the package along the run package installation path.  
  >   - If you select this parameter, .whl installs in the `${ascend_install_path}/cann/python/site-packages` path.
  >   - If you do not select this parameter, .whl installs in the local python path, for example, `/usr/local/python3.7.5/lib/python3.7/site-packages`.
  > - --autofuse indicates whether to install autofuse component-related compilation artifacts. Current package installation does not install autofuse by default. Add this option to install autofuse.
  > - For more installation options, use the --help option to view.  

- Uninstallation

  If you want to uninstall the installed package, execute the following command:

  ```shell
  # Add --install-path=${install_path} if installed to a specified path
  ./cann-graph-autofusion_${cann_version}_linux-${arch}.run --uninstall
  ```  

**After installation, refer to [Sample Execution](../../super_kernel/examples/README_en.md) to try running samples**.
