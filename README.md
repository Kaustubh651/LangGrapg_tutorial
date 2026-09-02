# LangGraph Tutorial

This repository contains hands-on notebooks for learning LangGraph and its
workflow patterns. Each section builds on the previous one, moving from basic
graphs to workflows that make decisions and improve their own output.

## Learning Path

### 1. Sequential Workflows

Learn how to connect steps in a fixed order.

- [Installation test](Video6_Squential_workflow/0_test_installation.ipynb)
- [BMI calculator](Video6_Squential_workflow/1_bmi_cal.ipynb)
- [Simple LLM workflow](Video6_Squential_workflow/2_simple_llm_workflow.ipynb)
- [Prompt chaining](Video6_Squential_workflow/3_prompt_chaining.ipynb)

### 2. Parallel Workflows

Run independent graph branches at the same time and combine their results.

- [Batsman workflow](Video7_Parallel_workflow/1_batsman_workflow.ipynb)
- [UPSC essay workflow](Video7_Parallel_workflow/2_upsc_essay_workflow.ipynb)

### 3. Conditional Workflows

Route execution to different nodes based on the current state.

- [Quadratic workflow](Video8_conditional_workflow/1_quadractic_workflow.ipynb)
- [Review and reply workflow](Video8_conditional_workflow/2_review_reply_workflow.ipynb)

### 4. Iterative Workflows

Generate, evaluate, and improve an output through repeated graph iterations.

- [X post generator](Video9_iterative_workflow/1_X_post_generator.ipynb)

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- [Ollama](https://ollama.com/) for the LLM notebooks

## Setup

Create and activate a virtual environment:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
pip install langgraph langchain-ollama pydantic jupyter
```

Start Ollama and download the model used by the notebooks:

```powershell
ollama pull qwen3:4b
```

Launch Jupyter:

```powershell
jupyter notebook
```

Open a notebook from the learning path and run its cells from top to bottom.

## Core LangGraph Concepts

The examples demonstrate how to:

- Define shared graph state with `TypedDict`
- Create nodes that read and update state
- Connect nodes with sequential and conditional edges
- Run parallel branches
- Compile and invoke a graph
- Use an evaluator and iteration limit in an improvement loop

## Notes

The LLM examples require Ollama to be running locally. Model responses can vary
with the installed model version and its configuration.

