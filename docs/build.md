# 源码构建

## 1. 环境准备

本项目支持源码编译，在源码编译前，需要确保已经安装CANN软件（Ascend-cann-toolkit和Ascend-cann-ops（可选）），若运行业务，还需要安装NPU驱动和固件。

软件安装方式请根据如下描述进行选择：

| 安装方式       | 说明                                                         | 使用场景                                                     |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 使用WebIDE安装 | WebIDE可提供在线直接运行的昇腾环境，当前可提供单机算力，默认安装最新商发版CANN软件包（目前是CANN 8.5.0）和固件/驱动包。 | 适用于没有昇腾设备的开发者。                                 |
| 手动安装软件包 | -                                                            | 适用有昇腾设备，想体验手动安装CANN包或体验最新master分支能力的开发者。 |

### 方式一：使用WebIDE安装

对于无环境的用户，可直接使用WebIDE开发平台，即“**算子一站式开发平台**”，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的软件包，无需手动安装。更多关于开发平台的介绍请参考[LINK](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击“`云开发`”按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="./figures/cloudIDE.png" alt="云平台"  width="750px" height="90px">

2. 根据页面提示创建并启动云开发环境，单击“`连接 > WebIDE`”进入算子一站式开发平台，开源项目的资源默认在`/mnt/workspace`目录下。

   <img src="./figures/webIDE.png" alt="云平台"  width="1000px" height="150px">

### 方式二：手动安装软件包

1. **安装驱动与固件（可选，仅运行业务依赖）**

    驱动与固件的下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。驱动与固件是运行样例依赖，若仅编译环境，可以不安装。

2. **安装CANN包**    

   **场景1：体验master版本能力或基于master版本进行开发**

     请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

     1. 安装CANN Toolkit开发套件包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # 安装命令
        /Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```
    
     2. 安装CANN ops算子包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

    - `${cann_version}`：表示CANN包版本号。
    - `${arch}`：表示CPU架构，如`aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径，需要与Toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。
    - `${soc_name}`表示NPU型号名称。

   **场景2：体验已发布版本能力或基于已发布版本进行开发**

    如果您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本CANN软件包（仅支持CANN 8.5.0及后续版本）进行安装。   

## 2. 环境验证

安装完CANN包后，需验证环境是否正常。

```bash
# 查看CANN Toolkit的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。WebIDE场景下，请将/usr/local替换为/home/developer。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# 查看CANN ops的version字段提供的版本信息（默认路径安装），<opsname>表示待查询的ops子包的名称，请用户根据实际安装路径替换。WebIDE场景下，请将/usr/local替换为/home/developer。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## 3. 环境变量配置

根据实际场景，选择合适的命令：

  ```bash
  # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}） 
  source /usr/local/Ascend/cann/set_env.sh
  # 指定路径安装
  source ${install_path}/cann/set_env.sh
  ```

## 4. 源码编译

### 4.1 下载源码

若您的编译环境可以访问网络，编译过程中将自动下载开源第三方软件，可以使用如下命令进行源码下载：

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
 ```

若您的编译环境无法访问网络，您需要通过下列步骤在联网环境中下载源码及开源软件压缩包，并手动上传至您的编译环境中：

1. 在联网环境中，进入[本项目主页](https://gitcode.com/cann/graph-autofusion), 通过`下载ZIP`或`clone`按钮，根据指导，完成源码下载。

2. 下载`makeself`第三方开源软件。

   | 开源软件 | 版本 | 下载地址 |
   |---|---|---|
   | makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |

3. 将源码及第三方开源软件上传到离线编译环境中，解压缩源码。

   ```bash
   # 解压缩源码
   unzip graph-autofusion.zip
   ```

4. 在源码同级目录创建open_source目录，将第三方开源软件压缩包放到open_source目录下解压缩后改名。

   ```bash
   # 创建open_source目录
   mkdir open_source
   # 将第三方开源软件压缩包放到open_source目录下
   mv makeself-release-2.5.0-patch1.tar.gz open_source/
   # 解压缩第三方开源软件压缩包并改名
   cd open_source
   tar -zxvf makeself-release-2.5.0-patch1.tar.gz && mv makeself-release-2.5.0 makeself
   ```

5. 完成后文件目录如下：

   ```bash
   ├── graph-autofusion           # 解压后的源码仓
   │  ├── cmake
   │  └── ...
   ├── open_source                # 三方开源软件
   │  └── makeself
   │     ├── COPYING
   │     ├── Makefile
   │     ├── README.md
   │     └── ...
   ```

### 4.2 安装依赖

#### 4.2.1 安装依赖

以下所列为源码编译用到的依赖，请注意版本要求：

- Python3 >= 3.8.0 (建议使用Python虚拟环境)

  1. 创建虚拟环境并激活：

     ```shell
     python3 -m venv venv
     source venv/bin/activate
     ```

  2. 安装依赖：

     ```shell
     cd graph-autofusion
     pip3 install -r super_kernel/requirements-dev.txt
     ```

- CMake >= 3.16.0  (建议使用3.20.0版本)

   ```shell
   # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
   sudo apt-get install cmake
   ```

#### 4.2.2 检查编译环境

环境准备完成后，建议执行环境检查脚本，确认当前环境是否满足编译要求。

```bash
bash scripts/check_env.sh
```

检查结果说明如下：

| 状态 | 含义 | 处理建议 |
|---|---|---|
| **[PASS]** | 检查通过 | 无需处理 |
| **[WARNING]** | 非关键依赖缺失或版本存在偏差 | 建议修复，不影响核心编译 |
| **[ERROR]** | 关键依赖缺失或版本不兼容 | 必须修复，否则无法编译 |

> [!NOTE] 注意
> 环境检查脚本中所有的检查项和版本约束严格来源于 docs/build.md 和 super_kernel/requirements-dev.txt，如文档和依赖更新，请同步修改[脚本](../scripts/check_env.sh)。

### 4.3 编译

进入源码仓根目录，执行以下命令进行编译：

```shell
bash build.sh --pkg
```

更多编译参数可以通过`bash build.sh -h`查看，执行成功后会在`build_out`目录下生成`cann-graph-autofusion_${cann_version}_linux-${arch}.run`。

- --pkg 表示构建 run 包。
- ${cann_version} 表示 cann 版本号。
- ${arch} 表示 CPU 架构，如 aarch64、x86_64。

### 4.4 测试验证

编译完成后，用户可以进行开发者测试，在执行本章节操作之前，确保已完成[环境准备](#环境准备)。

- UT 验证

   ```bash
   bash build.sh -u
   ```

   执行完成后根据输出日志查看UT测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

- ST 验证

   ```bash
   bash build.sh -s
   ```

   执行完成后根据输出日志查看ST测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

- 覆盖率验证

   ```bash
   bash build.sh -u -c  # UT 覆盖率
   bash build.sh -s -c  # ST 覆盖率
   bash build.sh -c     # UT 覆盖率 + ST 覆盖率
   ```

   执行完成后根据输出日志查看覆盖率情况，确认所有测试用例通过。

### 4.5 安装与卸载

- 安装

  本地验证完成后，可执行如下命令安装编译生成的软件包，执行安装命令时，请确保安装用户对软件包具有可执行权限。

  ```shell
  # 如果需要指定安装路径，则加上 --install-path=${install_path}
  ./cann-graph-autofusion_${cann_version}_linux-${arch}.run --full --quiet --pylocal
  ```

  > [!NOTE]说明
  > - 此处的安装路径（无论默认还是指定）需与前面安装 cann-toolkit 包时的路径保持一致。
  > - --full          全量模式安装。
  > - --install-path  指定安装路径，不指定则默认安装在`/usr/local/Ascend`（root 用户）或`${HOME}/Ascend`（非 root 用户）目录。
  > - --quiet         静默安装，跳过人机交互环节。  
  > - --pylocal       安装 run 包时，是否将包内的 .whl 跟随 run 包安装路径来安装。  
  >   - 若选择该参数，则 .whl 安装在`${ascend_install_path}/cann/python/site-packages`路径下。
  >   - 若不选择该参数，则 .whl 安装在本地 python 路径下，例如`/usr/local/python3.7.5/lib/python3.7/site-packages`。
  > - 更多安装选项请使用 --help 选项查看。  

- 卸载

  若您想卸载安装的软件包，可执行如下命令：

  ```shell
  # 如果是安装到指定路径情况，则加上 --install-path=${install_path}
  ./cann-graph-autofusion_${cann_version}_linux-${arch}.run --uninstall
  ```  

**安装完成后可参考[样例运行](../super_kernel/examples/README.md)尝试运行样例**。  
