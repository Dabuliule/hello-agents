from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from codecraft.skill import SkillMetadata
from codecraft.tui.commands import (
    ComposerChoice,
    ComposerMenuMode,
    command_choices,
    parse_composer_menu,
    skill_choices,
)
from codecraft.tui.theme import palette_for


class ComposerMenu(Vertical):
    """输入框上方的 slash command 与 Skill 候选列表。"""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._choices: dict[str, ComposerChoice] = {}

    def compose(self) -> ComposeResult:
        yield OptionList(
            id="composer-options",
            markup=False,
            compact=True,
        )

    def refresh_for(self, value: str, skills: tuple[SkillMetadata, ...]) -> bool:
        query = parse_composer_menu(value)
        if query is None:
            self.close()
            return False

        choices = (
            command_choices(query.query)
            if query.mode == ComposerMenuMode.COMMANDS
            else skill_choices(skills, query.query)
        )
        self._choices = {choice.id: choice for choice in choices}
        options = self.query_one(OptionList)
        if choices:
            options.set_options(
                Option(self._choice_prompt(choice), id=choice.id) for choice in choices
            )
            options.highlighted = 0
        else:
            label = (
                "No matching skills"
                if query.mode == ComposerMenuMode.SKILLS
                else "No matching commands"
            )
            options.set_options([Option(label, id="composer-empty", disabled=True)])
            options.highlighted = None
        self.display = True
        return True

    def close(self) -> None:
        self._choices.clear()
        self.display = False

    def move(self, offset: int) -> None:
        options = self.query_one(OptionList)
        if offset > 0:
            options.action_cursor_down()
        elif offset < 0:
            options.action_cursor_up()

    def selected_choice(self, choice_id: str | None = None) -> ComposerChoice | None:
        if choice_id is None:
            highlighted = self.query_one(OptionList).highlighted_option
            choice_id = highlighted.id if highlighted is not None else None
        return self._choices.get(choice_id or "")

    @staticmethod
    def insert_skill(value: str, skill_name: str) -> str | None:
        query = parse_composer_menu(value)
        if query is None:
            return None
        return (
            value[: query.replace_start]
            + f"${skill_name} "
            + value[query.replace_end :]
        )

    def _choice_prompt(self, choice: ComposerChoice) -> Text:
        palette = palette_for(self.app.current_theme.dark)
        prompt = Text(choice.title, style=f"bold {palette.strong}")
        prompt.append("  ")
        prompt.append(choice.description, style=palette.muted)
        return prompt
