# Environment Deployment

## 1. Environment Preparation

This project supports compilation from source. Before compiling from source, ensure you have installed CANN software (Ascend-cann-toolkit and Ascend-cann-ops (optional)). If you run samples, you also need to install NPU drivers and firmware.

Select the software installation method based on the following descriptions:

| Installation Method | Description | Use Case |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| WebIDE Installation | WebIDE provides an online directly runnable Ascend environment. It currently provides single-machine computing power. **It installs the latest commercial release CANN software package (currently CANN 8.5.0) and firmware/driver package by default.**<br>**For using the master branch, refer to Method 3 to install the latest community package.** | Suitable for developers without Ascend devices. |
| Docker | Docker images provide an efficient deployment method. One-click deployment of CANN packages and essential dependencies.<br>Currently, the OS only supports Ubuntu operating system. **It installs version 9.0.0 CANN software package and firmware/driver package by default.** | Suitable for developers with Ascend devices who need to quickly set up the environment. |
| Manual Package Installation | - | Suitable for developers with Ascend devices who want to experience manual CANN package installation or the latest master branch capabilities. |

### Method 1: Use WebIDE Installation

For users without an environment, use the WebIDE development platform directly. This "**one-stop operator development platform**" provides an online directly runnable Ascend environment. The environment has essential software packages pre-installed. No manual installation required. For more information about the development platform, refer to [LINK](https://gitcode.com/org/cann/discussions/54).

> **Note**:
The environment installs the latest commercial release CANN software package (currently CANN 8.5.0) and firmware/driver package by default. Pay attention to software compatibility when downloading source code.

1. Enter the open-source project. Click the "`Cloud Development`" button. Log in with an authenticated Huawei Cloud account. If not registered or authenticated, follow the page prompts to register and authenticate.

   <img src="../figures/cloudIDE.png" alt="Cloud Platform"  width="750px" height="90px">

2. Follow the page prompts to create and start the cloud development environment. Click "`Connect > WebIDE`" to enter the one-stop operator development platform. Open-source project resources default to the `/mnt/workspace` directory.

   <img src="../figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px">

### Method 2: Docker Deployment

For developers who do not depend on Ascend devices, if you want to quickly set up a compilation build environment, use Docker image deployment.

> **Note**: 
Image files are large. Downloading requires some time. Please wait patiently. For docker command option descriptions, query through `docker --help`.
<br>The environment installs version 9.0.0 CANN software package and firmware/driver package by default. Pay attention to software compatibility when downloading source code.

1.**Install Driver and Firmware (Runtime Dependency)**

For Ascend driver and firmware download and installation operations on the host machine, refer to the "Prepare Software Packages" and "Install NPU Driver and Firmware" chapters in the "[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)".

2.**Download Image**

- Step 1: Log in to the host machine as root user. Ensure Docker engine (version 1.11.2 or above) is installed on the host machine.
- Step 2: Pull the pre-integrated CANN software package and development-required dependencies image from the [Ascend Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). Use the following command. Select based on actual architecture (using Atlas A2 series products as example, compilation-only scenarios do not need attention):

    ```bash
    # Example: Pull ARM architecture CANN development image
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11
    # Example: Pull X86 architecture CANN development image
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11
    ```

3.**Run Docker**
Select different startup methods based on use case:

- **Scenario 1: Compilation Build Only (No Sample Execution Required)**

  If you only need code compilation build without NPU device access, use the following simplified command:

  ```bash
  docker run --name cann_container -it -u root --privileged=true -v /home/gaf/:/home/gaf swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11 bash
  ```

- **Scenario 2: Sample Execution Required (NPU Device Access Required)**

  If you need to run samples or tests, the container needs to access the host machine's NPU devices. Using Atlas A2 series products as example:

  ```bash
  docker run --name cann_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it -u root --privileged=true \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11 bash
  ```

  | Parameter | Description | Note |
  | :--- | :--- | :--- |
  | `--name cann_container` | Specifies a name for the container for management. | Optional (value can be customized). |
  | `--device /dev/davinci0` | Maps NPU device files into the container. | Required (when running samples). For multiple devices, use this parameter multiple times, for example, `/dev/davinci0` `/dev/davinci1`. |
  | `--device /dev/davinci_manager` | Ascend device manager, responsible for device resource management. | Required (when running samples). |
  | `--device /dev/devmm_svm` | Device memory management unit. | Required (when running samples). |
  | `--device /dev/hisi_hdc` | Ascend high-definition codec device. | Required (when running samples). |
  | `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts DCMI (Device Communication Management Interface) directory. | Required (when running samples). |
  | `-v /usr/local/bin/npu-smi:...` | Mounts NPU monitoring tool, used to view NPU status. | Required (when running samples). |
  | `-v /usr/local/Ascend/driver/...` | Mounts NPU driver libraries and version information. | Required (when running samples). |
  | `-v /etc/ascend_install.info:...` | Mounts Ascend software installation information. | Required (when running samples). |
  | `-it` | Combined parameter of `-i` (interactive) and `-t` (allocate pseudo-terminal). | Required |
  | `-u root` | Enter container as root (administrator). | Recommended |
  | `--privileged=true` | Enables container highest privilege mode. | Recommended |
  | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:...` | Specifies the Docker image to run. | Required. Ensure this image name and tag match exactly with the image you pulled through `docker pull`. |
  | `bash` | Command to execute immediately after container starts. | Required |

  > **Note**:
  > - Scenario 1 applies to compilation-only scenarios for graph-autofusion. No NPU device support required.
  > - Scenario 2 applies to scenarios requiring sample execution or NPU-related testing. Host machine requires NPU driver and firmware installed.
  > - If using other chip models (such as Ascend950PR or Atlas A3 series products), adjust device names in the `--device` parameter accordingly.

4.**Initialize Environment**
After entering the container, execute the following command to initialize the environment:

  ```bash
  curl -fsSL https://raw.gitcode.com/cann/graph-autofusion/raw/master/scripts/init_env.sh | bash
  ```

> **Note**:
> - CANN package and operator package installation occurs by default.
> - For other chip models, use the `--chip-type` parameter to specify the corresponding model (such as `950` or `A3`). For optional parameters, refer to [
init_env.sh](../../scripts/init_env.sh)

### Method 3: Manual Package Installation

1. **Install Driver and Firmware (Optional, Only Required for Running [Samples](../../autofuse/examples/README_en.md))**

    For driver and firmware download and installation operations, refer to the "Prepare Software Packages" and "Install NPU Driver and Firmware" chapters in the "[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)". Driver and firmware are dependencies for running samples. If you only compile the environment, you do not need to install them.

2. **Install CANN Package**    

   **Scenario 1: Experience master version capabilities or develop based on master version**

     Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) to get the latest time version. Download the corresponding package based on product model and environment architecture. Use the following installation command. For more guidance, refer to [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

     1. Install CANN Toolkit development kit package.

        ```bash
        # Ensure the installation package has execute permission
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        - `${cann_version}`: Indicates CANN package version number.
        - `${arch}`: Indicates CPU architecture, for example, `aarch64` or `x86_64`.
        - `${install_path}`: Indicates the specified installation path. Must match the Toolkit package installation path. Root user default installation path is `/usr/local/Ascend` directory.
    
     2. Install CANN ops operator package.

        ```bash
        # Ensure the installation package has execute permission
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        `${soc_name}` indicates the NPU model name.

   **Scenario 2: Experience released version capabilities or develop based on released version**

    If you want to experience **official released CANN package** capabilities, visit the [CANN Official Download Center](https://www.hiascend.com/cann/download). Select the corresponding version CANN software package (only supports CANN 8.5.0 and later versions) for installation.