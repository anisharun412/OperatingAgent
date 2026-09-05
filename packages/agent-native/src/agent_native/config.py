"""What an agent *is*.

An `AgentConfig` is the whole personality of one agent: which model it thinks
with, how many turns it may take, which tools it's allowed to touch, and the
instructions it starts from. Two agents ("build", "chat") differ only in their
config, not their code.

`PromptBuilder` turns that config plus the session (what folder are we in, what's
in it, what's today's date, which tools exist, what the user's standing
instructions are, what was learned in earlier conversations) into the single
system message the model reads first. Keeping prompt assembly in one place means
the model never gets a stale or hand-built prompt.

**Why the prompt names what's in the folder.** Told only "Working folder: .", a
model spends its first turn or two listing the folder and reading a manifest to
work out where it is - two round trips, paid for on every new session, to learn
something the machine already knows. Thirty names and a branch cost a few dozen
tokens once. This is a *shallow* look on purpose: the top level answers "what kind
of project is this", and anything deeper is a question the model should ask with a
tool, when it has a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: How many names from the working folder to put in the prompt. Enough to tell a
#: Python project from a website; not so many that a crowded folder pushes the
#: user's own standing instructions out of the model's attention.
MAX_LISTED_ENTRIES = 30

#: Names worth leaving out: caches, virtual environments and editor state. They
#: say nothing about the project and would fill the list on the way to saying it.
SKIPPED_NAMES = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        "target",
        "site-packages",
    }
)


@dataclass
class Subagent:
    """A named helper an agent can hand a smaller job to.

    Just a description - the same shape as a small AgentConfig. What runs one is
    the `delegate` tool in `tools/subagent.py`; a blank model or prompt here means
    "same as the agent that's delegating".
    """

    name: str
    description: str = ""
    model: str = ""
    system_prompt: str = ""


@dataclass
class AgentConfig:
    """Everything that makes one agent different from another."""

    name: str = "build"
    model: str = "gpt-oss-120b"
    #: Models to fall back to, in order, when `model` fails in a way that looks
    #: temporary - a rate limit, a provider outage - and retrying the same model
    #: hasn't cleared it. Empty is the norm, and a run with no fallbacks behaves
    #: exactly as it did before this field existed. A name that isn't registered
    #: is skipped rather than fatal, so listing an optional second provider here is
    #: safe. The mechanism lives in `AgentLoop._resolve_candidates` and `run_turn`.
    fallback_models: list = field(default_factory=list)
    description: str = ""
    system_prompt: str = "You are a careful, helpful assistant that can use tools."
    temperature: float = 0.0  # deterministic by default, so runs are reproducible
    top_p: float = 1.0
    max_output_tokens: int | None = None
    timeout_seconds: int = 60
    max_turns: int = 10
    #: The agent's default thinking budget: "" (let the provider decide) or one of
    #: "low"/"medium"/"high". A per-run `Limits.reasoning_effort` overrides it; a
    #: model with no thinking mode ignores it. It lives here as well as on Limits
    #: for the same reason `max_turns` does - the agent carries a sensible default
    #: that a single run can still widen or narrow without redefining the agent.
    reasoning_effort: str = ""
    allowed_tools: list = field(default_factory=list)  # empty means "all tools"
    subagents: list = field(default_factory=list)      # list[Subagent]


class PromptBuilder:
    """Turns an AgentConfig + Session into the text of the system message."""

    def build(
        self,
        config: AgentConfig,
        session: Any,
        tool_names: list | None = None,
        project_instructions: str = "",
        remembered: str = "",
        skills: str = "",
    ) -> str:
        lines = [config.system_prompt.strip(), ""]
        workdir = getattr(session, "working_directory", ".")
        lines.append(f"Working folder: {workdir}")
        lines.append(
            "Paths you give to file tools are relative to this working folder, so "
            'pass a bare name like "config.txt" - not the folder path - to reach a '
            "file inside it."
        )
        branch = read_branch(workdir)
        if branch:
            lines.append(f"Git branch: {branch}")
        listing = read_folder_listing(workdir)
        if listing:
            lines.append("At the top level of it: " + listing)
        lines.append(f"Today's date: {datetime.now(UTC):%Y-%m-%d}")
        if tool_names:
            lines.append("Tools you can use: " + ", ".join(tool_names) + ".")
        lines.append(
            "When you need a tool, call it. When a tool result comes back, read it "
            "and then either call another tool or give the user your final answer."
        )
        # A catalogue of skills sits above the standing instructions but below the
        # tool line: it's reference the model reaches for mid-task, not a rule to
        # obey. Only names and one-liners are here; a skill's body is loaded on
        # demand (the invoke_skill tool), which is the whole progressive-disclosure
        # economy - the listing costs a line, the body is paid for only when used.
        if skills:
            lines.extend(["", skills.strip()])
        # Both of these go last, and in this order, because they're the parts most
        # worth obeying: the user's own standing instructions for the project, then
        # what was learned in earlier conversations. The instructions outrank the
        # notes, since the user wrote them deliberately and by hand.
        if project_instructions:
            lines.extend(
                [
                    "",
                    (
                        "The user left standing instructions for this project. Follow "
                        "them over your own defaults:"
                    ),
                    project_instructions.strip(),
                ]
            )
        if remembered:
            lines.extend(
                [
                    "",
                    "From earlier conversations:",
                    remembered.strip(),
                    (
                        "If one of these is now wrong, say so and keep the corrected "
                        "version with the remember tool."
                    ),
                ]
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# What's in the folder
#
# Both of these are read once, when a session's prompt is built, and neither ever
# raises: a folder that can't be listed or a repo that can't be read means the
# prompt simply doesn't mention it, which is how the agent behaved before this
# existed. Neither one starts a subprocess - `git` may not be installed, and
# waiting on a child process to build a prompt is a bad trade for one branch name.
# ---------------------------------------------------------------------------
def read_folder_listing(working_directory: str, limit: int = MAX_LISTED_ENTRIES) -> str:
    """The names at the top of the working folder, folders first, as one line.

    Folders get a trailing slash so the model can tell them apart at a glance, and
    they come first because they're the part that says what kind of project this
    is. Hidden names are left out - a list led by `.gitignore` and `.env` says
    less about the project, and `.env` in particular is not a name to volunteer.
    """
    try:
        entries = sorted(
            (entry for entry in Path(working_directory).expanduser().iterdir()),
            key=lambda entry: entry.name.lower(),
        )
    except (OSError, RuntimeError, ValueError):
        return ""

    folders: list = []
    files: list = []
    for entry in entries:
        name = entry.name
        if name.startswith(".") or name in SKIPPED_NAMES:
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:  # a broken link, or a mount that went away
            continue
        (folders if is_dir else files).append(name + "/" if is_dir else name)

    names = folders + files
    if not names:
        return ""
    shown, hidden = names[:limit], len(names) - limit
    listing = ", ".join(shown)
    return listing + (f" (+{hidden} more)" if hidden > 0 else "")


def read_branch(working_directory: str) -> str:
    """The checked-out git branch, read straight out of `.git`. "" if there isn't one.

    A detached HEAD reports a short commit id instead, prefixed so it can't be
    mistaken for a branch name. Worktrees and submodules keep a `.git` *file*
    pointing elsewhere, which is followed once.
    """
    try:
        git = Path(working_directory).expanduser() / ".git"
        if git.is_file():  # a worktree or submodule: "gitdir: <path>"
            pointer = git.read_text(encoding="utf-8", errors="replace").strip()
            if not pointer.startswith("gitdir:"):
                return ""
            git = Path(pointer.split(":", 1)[1].strip())
            if not git.is_absolute():
                git = (Path(working_directory).expanduser() / git).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, RuntimeError, ValueError):
        return ""
    if head.startswith("ref: refs/heads/"):
        return head[len("ref: refs/heads/") :].strip()
    if head.startswith("ref: "):
        return head[len("ref: ") :].strip()
    return f"detached at {head[:8]}" if head else ""


# ---------------------------------------------------------------------------
# Skills
#
# A skill is a named folder of instructions the agent pulls in only when it's
# relevant - the progressive-disclosure pattern. Discovery is a folder-reading
# concern, so it lives here beside the folder listing and the branch read, and
# like them it never raises: a workspace with no skills, or an unreadable one,
# simply produces an empty catalogue, which is exactly how the agent behaved
# before skills existed. Only names and one-line descriptions are read here; a
# skill's full body is loaded on demand by the invoke_skill tool (tools/skill_tool.py).
# ---------------------------------------------------------------------------

#: A skill is a named folder holding this file - the same shape the harness uses.
SKILL_FILE = "SKILL.md"

#: Where skills live, relative to the working folder. Only the top level of each
#: root is scanned: a skill is a folder, not a tree to crawl.
SKILL_ROOTS = ("skills", ".agent/skills")

#: The listing sits in every prompt, so it's capped to stay a catalogue rather
#: than a chapter - enough skills to be useful, each description a line not a page.
MAX_SKILLS = 50
MAX_DESCRIPTION_CHARS = 200


@dataclass(frozen=True)
class Skill:
    """One discovered skill: how it's listed cheaply, and how its body is loaded."""

    name: str
    description: str
    path: str  # the SKILL.md file this was read from

    def body(self) -> str:
        """The skill's full instructions (frontmatter stripped), read fresh from disk.

        Read at invoke time, not discovery time - that's the whole point of
        progressive disclosure: the cheap listing costs a line, and the body is
        paid for only when the model actually reaches for the skill. Never raises;
        an unreadable file is an empty body.
        """
        try:
            text = Path(self.path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return ""
        _, body = _split_frontmatter(text)
        return body.strip() or text.strip()


def discover_skills(working_directory: str) -> list[Skill]:
    """Find every skill under the working folder's skill roots. Never raises.

    A skill is a named subfolder holding a ``SKILL.md``. Only the top level of
    each root is scanned, and only the name and description are read now - the
    body waits for an invoke. Duplicate names keep the first seen, and the result
    is sorted by name so a prompt built twice reads the same.
    """
    if not working_directory:
        return []
    base = Path(working_directory).expanduser()
    found: dict = {}
    for root_name in SKILL_ROOTS:
        try:
            subdirs = sorted(
                (entry for entry in (base / root_name).iterdir() if entry.is_dir()),
                key=lambda entry: entry.name.lower(),
            )
        except (OSError, RuntimeError, ValueError):
            continue  # a missing root is the normal case
        for sub in subdirs:
            manifest = sub / SKILL_FILE
            try:
                if not manifest.is_file():
                    continue
                text = manifest.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            meta, _ = _split_frontmatter(text)
            name = (meta.get("name") or sub.name).strip()
            raw_description = meta.get("description") or _first_meaningful_line(text)
            description = " ".join(raw_description.split())[:MAX_DESCRIPTION_CHARS]
            if name and name not in found:
                found[name] = Skill(name=name, description=description, path=str(manifest))
    return sorted(found.values(), key=lambda skill: skill.name.lower())[:MAX_SKILLS]


def skill_listing(skills: list[Skill]) -> str:
    """The cheap, always-in-the-prompt catalogue: one line per skill, bodies left out.

    This is the economy of the feature - the model learns a skill *exists* and what
    it's for from a line, and pays for the full instructions only if it calls
    ``invoke_skill``. Returns "" when there are no skills, so the prompt gains nothing.
    """
    if not skills:
        return ""
    lines = [
        (
            "Skills available for this project. Each is a set of instructions you load "
            "only when a task calls for it; when one matches, call the invoke_skill tool "
            "with its name to read its full instructions before you start:"
        )
    ]
    lines.extend(
        f"- {skill.name}: {skill.description}" if skill.description else f"- {skill.name}"
        for skill in skills
    )
    return "\n".join(lines)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a leading ``---`` frontmatter block into a dict, plus the body after it.

    Only the handful of ``key: value`` lines a skill needs (name, description) are
    read - this is deliberately not a YAML parser. A file with no frontmatter, or
    one whose fence never closes, comes back as ``({}, whole-text)``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict = {}
    for index in range(1, len(lines)):
        stripped = lines[index].strip()
        if stripped == "---":
            return meta, "\n".join(lines[index + 1 :])
        if ":" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(":")
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return {}, text  # no closing fence: treat the file as bodied, no metadata


def _first_meaningful_line(text: str) -> str:
    """The first real line of prose, for a skill that ships no description.

    Skips blank lines, the ``---`` fences, and a leading Markdown ``#`` so a skill
    whose first line is a heading still yields something readable.
    """
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if line and line != "---":
            return line
    return ""
