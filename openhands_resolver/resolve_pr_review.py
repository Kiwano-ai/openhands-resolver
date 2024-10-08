# flake8: noqa: E501

import asyncio
import dataclasses
import re
import shutil
from typing import Any, Awaitable, TextIO
import requests
import argparse
import litellm
import os
import pathlib
import subprocess
import jinja2

from tqdm import tqdm

from openhands_resolver.github_pr_review import GithubPRReview
from openhands_resolver.resolver_output import ResolverOutput
import openhands
from openhands.core.main import create_runtime, run_controller
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.memory.history import ShortTermHistory
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
)
from openhands.core.config import LLMConfig
from openhands.runtime.runtime import Runtime
from openhands_resolver.utils import (
    codeact_user_response,
    reset_logger_for_multiprocessing,
)

# Don't make this configurable for now, unless we have other competitive agents
AGENT_CLASS = "CodeActAgent"

def initialize_runtime(runtime: Runtime):
    """Initialize the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to change directory to /workspace.\n{obs}")

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config.\n{obs}")

async def complete_runtime(runtime: Runtime, base_commit: str) -> dict[str, Any]:
    """Complete the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to change directory to /workspace. Observation: {obs}")

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config. Observation: {obs}")

    action = CmdRunAction(command='git config --global --add safe.directory /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config. Observation: {obs}")

    action = CmdRunAction(command='git add -A')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to git add. Observation: {obs}")

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {base_commit}',
            keep_prompt=False,
        )
        action.timeout = 600 + 100 * n_retries
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                await asyncio.sleep(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            await asyncio.sleep(10)
        else:
            raise ValueError(f'Unexpected observation type: {type(obs)}')

    logger.info('-' * 30)
    logger.info('END Runtime Completion')
    logger.info('-' * 30)
    return {'git_patch': git_patch}

def get_instruction(pr_review: GithubPRReview, prompt_template: str):
    """Prepare instruction for the agent."""
    template = jinja2.Template(prompt_template)
    instruction = template.render(review_comments=pr_review.review_comments)
    return instruction

def guess_success(pr_review: GithubPRReview, history: ShortTermHistory, llm_config: LLMConfig) -> tuple[bool, str]:
    """Guess if the PR review comments have been addressed based on the history."""
    last_message = history.get_events_as_list()[-1].message
    
    prompt = f"""Given the following PR review comments and the last message from an AI agent attempting to address them, determine if the comments have been successfully resolved.

PR review comments:
{pr_review.review_comments}

Last message from AI agent:
{last_message}

(1) Have the PR review comments been successfully addressed?
(2) If the comments have been addressed, please provide an explanation of what was done in the PR that can be sent to the reviewer on GitHub. If the comments have not been addressed, please provide an explanation of why.

Answer in exactly the format below, with only true or false for success, and an explanation of the result.

--- success
true/false

--- explanation
...
"""

    response = litellm.completion(
        model=llm_config.model,
        messages=[{"role": "user", "content": prompt}],
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )
    
    answer = response.choices[0].message.content.strip()
    pattern = r'--- success\n*(true|false)\n*--- explanation*\n(.*)'
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        return match.group(1).lower() == 'true', match.group(2)
    else:
        return False, f"Failed to decode answer from LLM response: {answer}"

async def process_pr_review(
    pr_review: GithubPRReview,
    base_commit: str,
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    runtime_container_image: str,
    prompt_template: str,
) -> ResolverOutput:
    """Process a single PR review."""
    logger.info(f'Starting to address PR review comments for PR {pr_review.pr_number}.')

    workspace_base = os.path.join(output_dir, "workspace", f"pr_{pr_review.pr_number}")
    workspace_base = os.path.abspath(workspace_base)
    if os.path.exists(workspace_base):
        shutil.rmtree(workspace_base)
    shutil.copytree(os.path.join(output_dir, "repo"), workspace_base)

    config = AppConfig(
        default_agent="CodeActAgent",
        runtime='eventstream',
        max_budget_per_task=4,
        max_iterations=max_iterations,
        sandbox=SandboxConfig(
            runtime_container_image=runtime_container_image,
            enable_auto_lint=False,
            use_host_network=False,
            timeout=300,
        ),
        workspace_base=workspace_base,
        workspace_mount_path=workspace_base,
    )
    config.set_llm_config(llm_config)

    runtime = create_runtime(config, sid=f"pr_{pr_review.pr_number}")
    initialize_runtime(runtime)

    instruction = get_instruction(pr_review, prompt_template)

    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=codeact_user_response,
    )
    if state is None:
        raise RuntimeError("Failed to run the agent.")

    return_val = await complete_runtime(runtime, base_commit)
    git_patch = return_val['git_patch']
    logger.info(f'Got git diff for PR {pr_review.pr_number}:\n--------\n{git_patch}\n--------')

    histories = [dataclasses.asdict(event) for event in state.history.get_events()]
    metrics = state.metrics.get() if state.metrics else None

    success, success_explanation = guess_success(pr_review, state.history, llm_config)

    output = ResolverOutput(
        pr_review=pr_review,
        instruction=instruction,
        base_commit=base_commit,
        git_patch=git_patch,
        history=histories,
        metrics=metrics,
        success=success,
        success_explanation=success_explanation,
        error=state.last_error if state and state.last_error else None,
    )
    return output

async def resolve_pr_review(
    owner: str,
    repo: str,
    pr_number: int,
    review_id: int,
    token: str,
    max_iterations: int,
    output_dir: str,
    llm_config: LLMConfig,
    runtime_container_image: str,
    prompt_template: str,
) -> None:
    """Resolve PR review comments."""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    repo_dir = os.path.join(output_dir, "repo")
    if not os.path.exists(repo_dir):
        checkout_output = subprocess.check_output(
            [
                "git",
                "clone",
                f"https://x-access-token:{token}@github.com/{owner}/{repo}",
                f"{output_dir}/repo",
            ]
        ).decode("utf-8")
        if "fatal" in checkout_output:
            raise RuntimeError(f"Failed to clone repository: {checkout_output}")

    base_commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir
        )
        .decode("utf-8")
        .strip()
    )
    logger.info(f"Base commit: {base_commit}")

    output_file = os.path.join(output_dir, "output.jsonl")
    logger.info(f"Writing output to {output_file}")

    # Fetch PR review comments
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    review_comments = response.json()

    pr_review = GithubPRReview(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        review_id=review_id,
        review_comments=review_comments,
    )

    output = await process_pr_review(
        pr_review,
        base_commit,
        max_iterations,
        llm_config,
        output_dir,
        runtime_container_image,
        prompt_template,
    )

    with open(output_file, "a") as f:
        f.write(output.model_dump_json() + "\n")

    logger.info("Finished resolving PR review comments.")

def main():
    parser = argparse.ArgumentParser(description="Resolve PR review comments.")
    parser.add_argument("--repo", type=str, required=True, help="Github repository in form of `owner/repo`.")
    parser.add_argument("--pr-number", type=int, required=True, help="PR number to resolve.")
    parser.add_argument("--review-id", type=int, required=True, help="Review ID to resolve.")
    parser.add_argument("--token", type=str, default=None, help="Github token to access the repository.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum number of iterations to run.")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory to write the results.")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model to use.")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key to use.")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL to use.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to the prompt template file in Jinja format.")
    parser.add_argument("--runtime-container-image", type=str, default=None, help="Container image to use.")

    args = parser.parse_args()

    owner, repo = args.repo.split("/")
    token = args.token if args.token else os.getenv("GITHUB_TOKEN")

    if not token:
        raise ValueError("Github token is required.")

    llm_config = LLMConfig(
        model=args.llm_model or os.environ["LLM_MODEL"],
        api_key=args.llm_api_key or os.environ["LLM_API_KEY"],
        base_url=args.llm_base_url or os.environ.get("LLM_BASE_URL", None),
    )

    prompt_file = args.prompt_file
    if prompt_file is None:
        prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/pr-review.jinja")
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    runtime_container_image = f"ghcr.io/all-hands-ai/runtime:oh_v{openhands.__version__}_image_nikolaik_s_python-nodejs_tag_python3.11-nodejs22"

    asyncio.run(
        resolve_pr_review(
            owner=owner,
            repo=repo,
            pr_number=args.pr_number,
            review_id=args.review_id,
            token=token,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
            llm_config=llm_config,
            runtime_container_image=runtime_container_image,
            prompt_template=prompt_template,
        )
    )

if __name__ == "__main__":
    main()
