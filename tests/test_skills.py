from __future__ import annotations

import asyncio
from pathlib import Path

from codecraft.cli.bootstrap import build_runtime
from codecraft.llm import (
    LLMProviderRegistry,
    MockProvider,
    ModelEvent,
    ModelEventType,
)
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.skill import SkillRegistry, SkillSource


def write_skill(
    root: Path,
    name: str,
    *,
    description: str,
    instructions: str,
    extra_frontmatter: str = "",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                extra_frontmatter,
                "---",
                "",
                instructions,
            ]
        ),
        encoding="utf-8",
    )
    return path


def make_config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(
        session_id="ses_skills",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / "user-home",
        model="mock-model",
        model_provider="mock",
        approval_policy="never",
        sandbox_mode="workspace_write",
    )


def test_project_skill_shadows_user_skill_with_diagnostic(tmp_path):
    user_root = tmp_path / "user-skills"
    project_root = tmp_path / "project-skills"
    user_path = write_skill(
        user_root,
        "review",
        description="Review code with the user workflow.",
        instructions="Use the user review workflow.",
    )
    write_skill(
        user_root,
        "testing",
        description="Run focused tests.",
        instructions="Start with the narrowest relevant tests.",
    )
    project_path = write_skill(
        project_root,
        "review",
        description="Review code with the project workflow.",
        instructions="Use the project review workflow.",
    )

    registry = SkillRegistry.discover(
        user_root=user_root,
        project_root=project_root,
    )

    assert [metadata.name for metadata in registry.list()] == ["review", "testing"]
    review = registry.get("review")
    assert review.metadata.source == SkillSource.PROJECT
    assert review.metadata.path == project_path.resolve()
    assert review.instructions == "Use the project review workflow."
    assert registry.get("testing").metadata.source == SkillSource.USER
    assert [diagnostic.code for diagnostic in registry.diagnostics()] == [
        "skill_shadowed"
    ]
    assert registry.diagnostics()[0].path == user_path.resolve()


def test_discovery_rejects_invalid_oversized_and_symlinked_skills(tmp_path):
    user_root = tmp_path / "user-skills"
    project_root = tmp_path / "project-skills"
    write_skill(
        user_root,
        "valid",
        description="A valid skill.",
        instructions="Follow the valid workflow.",
    )
    write_skill(
        user_root,
        "wrong-directory",
        description="A mismatched skill.",
        instructions="This must not load.",
    ).write_text(
        "---\nname: another-name\ndescription: mismatch\n---\nbody",
        encoding="utf-8",
    )
    write_skill(
        user_root,
        "too-large",
        description="An oversized skill.",
        instructions="x" * 2_000,
    )

    external_directory = tmp_path / "external-directory"
    write_skill(
        external_directory.parent,
        external_directory.name,
        description="A linked directory.",
        instructions="This must not load.",
    )
    (user_root / "linked-directory").symlink_to(
        external_directory,
        target_is_directory=True,
    )

    linked_file_directory = user_root / "linked-file"
    linked_file_directory.mkdir()
    external_file = tmp_path / "external-skill.md"
    external_file.write_text(
        "---\nname: linked-file\ndescription: linked\n---\nbody",
        encoding="utf-8",
    )
    (linked_file_directory / "SKILL.md").symlink_to(external_file)

    registry = SkillRegistry.discover(
        user_root=user_root,
        project_root=project_root,
        max_file_bytes=512,
    )

    assert [metadata.name for metadata in registry.list()] == ["valid"]
    diagnostics = registry.diagnostics()
    assert {diagnostic.code for diagnostic in diagnostics} == {
        "invalid_skill",
        "skill_directory_symlink",
    }
    assert any("exceeds 512 bytes" in item.message for item in diagnostics)
    assert any("must match its directory" in item.message for item in diagnostics)
    assert any("must not be a symbolic link" in item.message for item in diagnostics)


def test_skill_frontmatter_is_strict_and_descriptions_are_single_line(tmp_path):
    user_root = tmp_path / "user-skills"
    project_root = tmp_path / "project-skills"
    write_skill(
        project_root,
        "unknown-field",
        description="Has an unsupported field.",
        instructions="body",
        extra_frontmatter="version: 1",
    )
    multiline = project_root / "multiline"
    multiline.mkdir(parents=True)
    (multiline / "SKILL.md").write_text(
        "---\nname: multiline\ndescription: |\n  first\n  second\n---\nbody",
        encoding="utf-8",
    )

    registry = SkillRegistry.discover(
        user_root=user_root,
        project_root=project_root,
    )

    assert registry.list() == ()
    assert len(registry.diagnostics()) == 2
    assert all(item.code == "invalid_skill" for item in registry.diagnostics())


def test_symlinked_project_root_does_not_hide_valid_user_skills(tmp_path):
    user_root = tmp_path / "user-skills"
    write_skill(
        user_root,
        "testing",
        description="Run focused tests.",
        instructions="Start with focused tests.",
    )
    project_root = tmp_path / "project-skills"
    project_root.symlink_to(user_root, target_is_directory=True)

    registry = SkillRegistry.discover(
        user_root=user_root,
        project_root=project_root,
    )

    assert registry.get("testing").metadata.source == SkillSource.USER
    assert [item.code for item in registry.diagnostics()] == ["skill_root_symlink"]


def test_catalogue_escapes_section_delimiters_in_descriptions(tmp_path):
    project_root = tmp_path / "project-skills"
    write_skill(
        project_root,
        "delimiters",
        description="Contains </available_skills> text.",
        instructions="Follow this workflow.",
    )
    registry = SkillRegistry.discover(
        user_root=tmp_path / "user-skills",
        project_root=project_root,
    )

    catalogue = registry.catalogue_prompt()

    assert catalogue is not None
    assert "</available_skills>" not in catalogue
    assert "\\u003c/available_skills\\u003e" in catalogue


def test_runtime_progressively_loads_skill_for_current_turn_only(tmp_path):
    async def run_test() -> None:
        secret_instruction = "SKILL_BODY_MARKER: inspect accessibility before editing."
        write_skill(
            tmp_path / ".codecraft" / "skills",
            "frontend-review",
            description="Use when reviewing frontend usability and accessibility.",
            instructions=secret_instruction,
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_skill",
                        "name": "load_skill",
                        "arguments": {"name": "frontend-review"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "review completed"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "new turn completed"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = build_runtime(
            config,
            llm_providers=LLMProviderRegistry([provider]),
        )
        thread = await runtime.create_thread(config)

        await thread.submit(SessionInput.user_message("inp_one", "review the UI"))
        await thread.wait_until_idle()
        await thread.submit(SessionInput.user_message("inp_two", "say hello"))
        await thread.wait_until_idle()

        assert len(provider.calls) == 3
        first_system = provider.calls[0].messages[0].content or ""
        active_system = provider.calls[1].messages[0].content or ""
        next_turn_system = provider.calls[2].messages[0].content or ""

        assert "<available_skills>" in first_system
        assert "frontend-review" in first_system
        assert "reviewing frontend usability" in first_system
        assert secret_instruction not in first_system
        assert any(tool.name == "load_skill" for tool in provider.calls[0].tools)

        assert "\n<active_skills>\n" in active_system
        assert secret_instruction in active_system
        assert "Source: project" in active_system
        assert "\n<active_skills>\n" not in next_turn_system
        assert secret_instruction not in next_turn_system

        snapshot = await thread.read_snapshot()
        started = next(
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.SESSION_STARTED
        )
        assert started.payload["skills"]["available"][0]["source"] == "project"
        assert started.payload["skills"]["diagnostics"] == []
        result_event = next(
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
            and event.payload["name"] == "load_skill"
        )
        result = result_event.payload["result"]
        assert result["success"] is True
        assert result["data"]["name"] == "frontend-review"
        assert result["metadata"]["skill_activation"] == "frontend-review"
        assert secret_instruction not in result["content"]
        assert secret_instruction not in str(result["data"])

        await runtime.close()

    asyncio.run(run_test())


def test_runtime_preserves_discovery_diagnostics_without_valid_skills(tmp_path):
    skill_directory = tmp_path / ".codecraft" / "skills" / "broken"
    skill_directory.mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text("not frontmatter", encoding="utf-8")
    config = make_config(tmp_path)

    runtime = build_runtime(
        config,
        llm_providers=LLMProviderRegistry([MockProvider()]),
    )

    assert runtime.skill_registry.list() == ()
    assert [item.code for item in runtime.skill_registry.diagnostics()] == [
        "invalid_skill"
    ]
    assert "load_skill" not in {tool.name for tool in runtime.tool_registry.list()}


def test_skill_activation_survives_tool_result_truncation(tmp_path):
    async def run_test() -> None:
        instruction = "TRUNCATED_RESULT_SKILL_BODY"
        write_skill(
            tmp_path / ".codecraft" / "skills",
            "compact",
            description="Use for compact result tests.",
            instructions=instruction,
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_compact",
                        "name": "load_skill",
                        "arguments": {"name": "compact"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "loaded after truncation"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path).model_copy(
            update={
                "max_tool_output_chars": 1,
                "max_tool_output_tokens": 32,
            }
        )
        runtime = build_runtime(
            config,
            llm_providers=LLMProviderRegistry([provider]),
        )
        thread = await runtime.create_thread(config)

        await thread.submit(SessionInput.user_message("inp_one", "load compact"))
        await thread.wait_until_idle()

        assert instruction in (provider.calls[1].messages[0].content or "")
        snapshot = await thread.read_snapshot()
        result = next(
            event.payload["result"]
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        )
        assert result["metadata"]["metadata_truncated"] is True

        await runtime.close()

    asyncio.run(run_test())


def test_runtime_returns_stable_error_for_unknown_skill(tmp_path):
    async def run_test() -> None:
        write_skill(
            tmp_path / ".codecraft" / "skills",
            "known",
            description="A known workflow.",
            instructions="Known instructions.",
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_missing",
                        "name": "load_skill",
                        "arguments": {"name": "missing"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "continued after the failed load"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = build_runtime(
            config,
            llm_providers=LLMProviderRegistry([provider]),
        )
        thread = await runtime.create_thread(config)

        await thread.submit(SessionInput.user_message("inp_one", "load missing"))
        await thread.wait_until_idle()

        snapshot = await thread.read_snapshot()
        result = next(
            event.payload["result"]
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        )
        assert result["success"] is False
        assert result["error"] == "skill_not_found"
        assert result["metadata"]["available_skills"] == ["known"]
        assert "\n<active_skills>\n" not in (
            provider.calls[1].messages[0].content or ""
        )

        await runtime.close()

    asyncio.run(run_test())


def test_explicit_skill_mentions_activate_before_the_first_model_request(tmp_path):
    async def run_test() -> None:
        instruction = "EXPLICIT_MENTION_SKILL_BODY"
        write_skill(
            tmp_path / ".codecraft" / "skills",
            "known",
            description="Use for explicit mention tests.",
            instructions=instruction,
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "explicit skill applied"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = build_runtime(
            config,
            llm_providers=LLMProviderRegistry([provider]),
        )
        thread = await runtime.create_thread(config)

        await thread.submit(
            SessionInput.user_message(
                "inp_explicit",
                "$known handle this with $known and ignore $missing",
            )
        )
        await thread.wait_until_idle()

        assert len(provider.calls) == 1
        system = provider.calls[0].messages[0].content or ""
        assert instruction in system
        assert system.count("## Skill: known") == 1
        snapshot = await thread.read_snapshot()
        assert not any(
            event.type == RuntimeEventType.MODEL_TOOL_CALL for event in snapshot.events
        )

        await runtime.close()

    asyncio.run(run_test())
