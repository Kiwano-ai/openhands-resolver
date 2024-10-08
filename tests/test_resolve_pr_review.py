import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from openhands_resolver.resolve_pr_review import (
    initialize_runtime,
    complete_runtime,
    get_instruction,
    process_pr_review,
    guess_success,
    resolve_pr_review,
)
from openhands_resolver.github_pr_review import GithubPRReview
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, NullObservation
from openhands_resolver.resolver_output import ResolverOutput
from openhands.core.config import LLMConfig
from openhands.memory.history import ShortTermHistory


@pytest.fixture
def mock_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = os.path.join(temp_dir, "repo")
        os.makedirs(repo_path)
        os.system(f"git init {repo_path}")
        readme_path = os.path.join(repo_path, "README.md")
        with open(readme_path, "w") as f:
            f.write("hello world")
        os.system(f"git -C {repo_path} add README.md")
        os.system(f"git -C {repo_path} commit -m 'Initial commit'")
        yield temp_dir


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.check_output") as mock_check_output:
        yield mock_check_output


@pytest.fixture
def mock_os():
    with patch("os.system") as mock_system, patch("os.path.join") as mock_join:
        yield mock_system, mock_join


@pytest.fixture
def mock_prompt_template():
    return "PR Review: {{ review_comments }}\n\nPlease address these comments."


def create_cmd_output(exit_code: int, content: str, command_id: int, command: str):
    return CmdOutputObservation(
        exit_code=exit_code, content=content, command_id=command_id, command=command
    )


def test_initialize_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(exit_code=0, content="", command_id=1, command="cd /workspace"),
        create_cmd_output(exit_code=0, content="", command_id=2, command='git config --global core.pager ""'),
    ]

    initialize_runtime(mock_runtime)

    assert mock_runtime.run_action.call_count == 2
    mock_runtime.run_action.assert_any_call(CmdRunAction(command="cd /workspace"))
    mock_runtime.run_action.assert_any_call(CmdRunAction(command='git config --global core.pager ""'))


@pytest.mark.asyncio
async def test_complete_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(exit_code=0, content="", command_id=1, command="cd /workspace"),
        create_cmd_output(exit_code=0, content="", command_id=2, command='git config --global core.pager ""'),
        create_cmd_output(exit_code=0, content="", command_id=3, command='git config --global --add safe.directory /workspace'),
        create_cmd_output(exit_code=0, content="", command_id=4, command='git add -A'),
        create_cmd_output(exit_code=0, content="git diff content", command_id=5, command="git diff --no-color --cached base_commit_hash"),
    ]

    result = await complete_runtime(mock_runtime, "base_commit_hash")

    assert result == {"git_patch": "git diff content"}
    assert mock_runtime.run_action.call_count == 5


def test_get_instruction(mock_prompt_template):
    pr_review = GithubPRReview(
        owner="test_owner",
        repo="test_repo",
        pr_number=1,
        review_id=123,
        review_comments=[{"body": "Please fix this", "path": "file.py", "position": 10}],
    )
    instruction = get_instruction(pr_review, mock_prompt_template)
    expected_instruction = "PR Review: [{'body': 'Please fix this', 'path': 'file.py', 'position': 10}]\n\nPlease address these comments."
    assert instruction == expected_instruction


@pytest.mark.asyncio
async def test_process_pr_review(mock_output_dir, mock_prompt_template):
    mock_create_runtime = AsyncMock()
    mock_initialize_runtime = AsyncMock()
    mock_run_controller = AsyncMock()
    mock_complete_runtime = AsyncMock()
    mock_guess_success = MagicMock()

    pr_review = GithubPRReview(
        owner="test_owner",
        repo="test_repo",
        pr_number=1,
        review_id=123,
        review_comments=[{"body": "Please fix this", "path": "file.py", "position": 10}],
    )
    base_commit = "abcdef1234567890"
    max_iterations = 5
    llm_config = LLMConfig(model="test_model", api_key="test_api_key")
    runtime_container_image = "test_image:latest"

    mock_create_runtime.return_value = MagicMock()
    mock_run_controller.return_value = MagicMock(
        history=MagicMock(get_events=MagicMock(return_value=[NullObservation(content="")])),
        metrics=MagicMock(get=MagicMock(return_value={"test_result": "passed"})),
        last_error=None,
    )
    mock_complete_runtime.return_value = {"git_patch": "test patch"}
    mock_guess_success.return_value = (True, "PR review comments addressed successfully")

    with patch("openhands_resolver.resolve_pr_review.create_runtime", mock_create_runtime), \
         patch("openhands_resolver.resolve_pr_review.initialize_runtime", mock_initialize_runtime), \
         patch("openhands_resolver.resolve_pr_review.run_controller", mock_run_controller), \
         patch("openhands_resolver.resolve_pr_review.complete_runtime", mock_complete_runtime), \
         patch("openhands_resolver.resolve_pr_review.guess_success", mock_guess_success), \
         patch("openhands_resolver.resolve_pr_review.logger"):

        result = await process_pr_review(
            pr_review,
            base_commit,
            max_iterations,
            llm_config,
            mock_output_dir,
            runtime_container_image,
            mock_prompt_template,
        )

        assert isinstance(result, ResolverOutput)
        assert result.pr_review == pr_review
        assert result.issue is None
        assert result.base_commit == base_commit
        assert result.git_patch == "test patch"
        assert result.success
        assert result.success_explanation == "PR review comments addressed successfully"
        assert result.error is None

        mock_create_runtime.assert_called_once()
        mock_initialize_runtime.assert_called_once()
        mock_run_controller.assert_called_once()
        mock_complete_runtime.assert_called_once()
        mock_guess_success.assert_called_once()


def test_guess_success():
    mock_pr_review = GithubPRReview(
        owner="test_owner",
        repo="test_repo",
        pr_number=1,
        review_id=123,
        review_comments=[{"body": "Please fix this", "path": "file.py", "position": 10}],
    )
    mock_history = MagicMock(spec=ShortTermHistory)
    mock_history.get_events_as_list.return_value = [
        create_cmd_output(exit_code=0, content="", command_id=1, command="cd /workspace")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="--- success\ntrue\n--- explanation\nPR review comments addressed successfully"))]
    
    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, explanation = guess_success(mock_pr_review, mock_history, mock_llm_config)
        assert success
        assert explanation == "PR review comments addressed successfully"


@pytest.mark.asyncio
async def test_resolve_pr_review(mock_output_dir):
    mock_process_pr_review = AsyncMock()
    mock_process_pr_review.return_value = ResolverOutput(
        pr_review=GithubPRReview(owner="test_owner", repo="test_repo", pr_number=1, review_id=123, review_comments=[]),
        instruction="test instruction",
        base_commit="abcdef1234567890",
        git_patch="test patch",
        history=[],
        metrics={},
        success=True,
        success_explanation="PR review comments addressed successfully",
        error=None,
    )

    with patch("openhands_resolver.resolve_pr_review.process_pr_review", mock_process_pr_review), \
         patch("openhands_resolver.resolve_pr_review.requests.get") as mock_get, \
         patch("builtins.open", MagicMock()):
        
        mock_get.return_value.json.return_value = [{"body": "Please fix this", "path": "file.py", "position": 10}]
        mock_get.return_value.raise_for_status = MagicMock()

        await resolve_pr_review(
            owner="test_owner",
            repo="test_repo",
            pr_number=1,
            review_id=123,
            token="test_token",
            max_iterations=5,
            output_dir=mock_output_dir,
            llm_config=LLMConfig(model="test_model", api_key="test_api_key"),
            runtime_container_image="test_image:latest",
            prompt_template="test template",
        )

        mock_process_pr_review.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/test_owner/test_repo/pulls/1/reviews/123/comments",
            headers={
                "Authorization": "token test_token",
                "Accept": "application/vnd.github.v3+json",
            },
        )


if __name__ == "__main__":
    pytest.main()
