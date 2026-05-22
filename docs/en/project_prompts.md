# Project Prompts

## File Description

This file stores **AI-assisted development prompts related to the Graph-autofusion project**.  
You can copy these prompts directly to VSCode Chat (GitHub Copilot Chat / ChatGPT Codex) for use.  
They help developers quickly initialize environments, run tests, or collaborate with intelligent agents.

> Note:
> - This file only records reusable intelligent agent prompts. It does not execute during project build or runtime.
> - Agents do not automatically read this file. They load only when manually referenced (for example, typing "read #file:doc/project_prompts.md").
> - When the project adds new development workflows, test tasks, or components (such as superkernel or autofuse-core), append corresponding prompt templates to this file.

---

## Environment Initialization and Testing Prompt

This prompt helps quickly initialize the Graph-autofusion project development environment in VSCode Copilot / ChatGPT Codex.  
It also verifies whether pytest tests pass correctly.

```prompt
# Graph-autofusion Project Development Environment Initialization

Read the following files to understand the project structure and development workflow:
- #file:README.md
- #file:developer_guide.md

Set environment variables:
  ASCEND_INSTALL_PATH=<!!!Configure your environment variable!!!>

Use the Python virtual environment in the project root directory:
  venv

---

## Initial Task

1. **Read and understand the project structure and dependencies**  
2. **Execute pytest tests** (run `pytest` in project root)  
   - If tests fail, list failed test items, error stack (traceback), and possible causes.  
   - Do not modify any code.  
3. **Report results** (summarize test results)  
   - When successful, state the number of tests and execution time.  
   - When failed, focus on possible dependency or environment issues.

---

## Next Step

After all pytest tests pass, we will:
- Develop and debug the **superkernel** component together;
- Optimize automatic fusion logic based on test results;
- During this process, you maintain Chinese communication, and I will direct the next development content.

---

## Rules
- Execute all system commands (such as `pytest`, `pip install`, `python setup.py build`) in the project root directory.  
- Modify source code only when I explicitly instruct.  
- Use Chinese for output reports. Keep code and commands in English.  

```