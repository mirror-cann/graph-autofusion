# ascir_tool

- **Configure Environment Variables**
  ```bash
  source env.sh
  ```

- **Execution**
* mode=0
  Single kernel end-to-end execution, including code generation and compilation based on input_ascir.py composition script, launching kernel according to ascir.json, generating input and gold data according to gen_input.py, and verifying kernel output against gold data.
  (1) Create use case directory under testcase directory (or other directory), create input_ascir.py, ascir.json, gen_input.py respectively.

  (2) Execute:
  ```bash
  bash test_ascir.sh --mode=0 --case=your_use_case_name_under_testcase_directory (--path=use_case_path)
  ```
  If profiling is needed, execute:
  ```bash
  bash test_ascir.sh your_use_case_name --prof
  ```

  (3) Verified use cases can be submitted to repository.

* mode=1
  Compile, execute, and compare results based on existing host and device code.
  (1) This scenario uses configuration information ascir.json under config path and gen_input.py under input path.

  (2) Execute:
  ```bash
  bash test_ascir.sh --mode=1 --path=/kernel_meta/build/
  ```
  The --path parameter is the kernel path. The path scenario is the build path for generated kernels when auto-fusion debug is enabled. Under build path, there are "host" and "device" folders, saving tiling code and kernel code respectively.

* mode=2
  Only compile based on existing host and device code.
  (1) Compile based on device kernel code.

  (2) Execute:
  ```bash
  bash test_ascir.sh --mode=2 --path=/kernel_meta/build/
  ```