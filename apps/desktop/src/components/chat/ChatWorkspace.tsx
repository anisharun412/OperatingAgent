import { useCallback, useEffect, useRef, useState } from "react";
import { activitySignature, isActivityEvent, nativeApi, readSSEStream, sseSubscribe, taskApi } from "../../lib/api";
import { folderName, isTauri, pickDirectory } from "../../lib/pickFolder";
import type { EventResponse, PermissionResponse, SessionResponse, ThreadResponse } from "../../lib/types";
import { loadSettings, saveSettings } from "../SettingsModal";
import { AlertDialog, ConfirmDialog, PromptDialog } from "../Modal";
import { AnalyticsView } from "./AnalyticsView";
import { MarkdownText } from "../MarkdownText";
import { formatLocalTime, timeAgo } from "../../lib/time";

type ChatItem =
  | { kind: "native"; id: string; title: string; subtitle: string; updatedAt: string }
  | { kind: "langgraph"; id: string; title: string; subtitle: string; updatedAt: string };

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "tool";
  text: string;
  parts?: unknown[];
  model?: string;
  time?: string;
  toolCalls?: Array<{ name: string; status: string; preview?: string }>;
  thinking?: string;
  streaming?: boolean;
};

const ACTIVE_TASK_STATUSES = new Set(["created", "pending", "running"]);
const TERMINAL_TASK_STATUSES = new Set(["completed", "failed", "interrupted"]);

function messageContentOf(value: Record<string, unknown>): unknown {
  const data = value.data;
  if (data && typeof data === "object") return (data as Record<string, unknown>).content;
  return value.content;
}

function isAiMessage(value: Record<string, unknown>): boolean {
  const type = String(value.type || "").toLowerCase();
  return type === "ai" || type === "aimessage" || type === "assistant";
}

/** Answer text from content that may be a string or a list of blocks. */
function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const record = block as Record<string, unknown>;
    const kind = String(record.type || "").toLowerCase();
    // Thinking/reasoning blocks are extracted separately; only "text" (and
    // untyped) blocks belong to the visible answer.
    if ((kind === "text" || kind === "") && typeof record.text === "string") {
      parts.push(record.text);
    }
  }
  return parts.join("");
}

function assistantTextFromState(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const messages = (payload as Record<string, unknown>).messages;
  if (!Array.isArray(messages)) return "";
  // Only the current turn counts. State is restored from the thread
  // checkpoint, so everything before the newest user message is history: the
  // last AI message there is the *previous* response, and returning it would
  // paint the previous answer into the live bubble while the agent works.
  let start = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") continue;
    const type = String((message as Record<string, unknown>).type || "").toLowerCase();
    if (type === "human" || type === "humanmessage" || type === "user") {
      start = index + 1;
      break;
    }
  }
  for (let index = messages.length - 1; index >= start; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") continue;
    const value = message as Record<string, unknown>;
    if (!isAiMessage(value)) continue;
    const text = textOfContent(messageContentOf(value));
    if (text.trim()) return text;
  }
  return "";
}

/** Reasoning/thinking text from provider-specific shapes, for live display. */
function reasoningTextFromState(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const messages = (payload as Record<string, unknown>).messages;
  if (!Array.isArray(messages)) return "";
  const parts: string[] = [];
  const push = (value: unknown) => {
    if (typeof value === "string" && value.trim()) parts.push(value);
  };
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const value = message as Record<string, unknown>;
    if (!isAiMessage(value)) continue;
    const data = value.data;
    const record = data && typeof data === "object" ? (data as Record<string, unknown>) : value;
    const extra = record.additional_kwargs;
    if (extra && typeof extra === "object") {
      const kwargs = extra as Record<string, unknown>;
      push(kwargs.reasoning_content);
      push(kwargs.reasoning);
      push(kwargs.thinking);
    }
    push(record.reasoning_content);
    push(record.reasoning);
    const content = record.content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== "object") continue;
        const item = block as Record<string, unknown>;
        const kind = String(item.type || "").toLowerCase();
        if (kind === "text" || kind === "") continue;
        push(item.text);
        push(item.reasoning);
        push(item.thinking);
      }
    }
  }
  return parts.join("\n");
}

type DialogState =
  | { kind: "rename"; current: string }
  | { kind: "delete"; id: string; chatKind: "native" | "langgraph"; title: string }
  | { kind: "fork" }
  | { kind: "error"; message: string }
  | null;

function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  if (!text && !live) return null;
  return (
    <div className="mb-2 rounded-lg overflow-hidden" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)", borderLeft: "2px solid var(--accent)" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left btn-quiet"
      >
        {live ? (
          <span className="w-1.5 h-1.5 rounded-full anim-pulse-dot shrink-0" style={{ background: "var(--accent)" }} />
        ) : (
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0" style={{ color: "var(--fg-3)", transform: open ? "rotate(180deg)" : undefined, transition: "transform var(--dur-1) ease" }}>
            <path d="m6 9 6 6 6-6" />
          </svg>
        )}
        <span className="text-[11px] font-medium" style={{ color: "var(--fg-2)" }}>
          {live ? "Thinking…" : "Thought process"}
        </span>
      </button>
      {open && (
        <div className={`px-2.5 pb-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words anim-fade-in ${live ? "stream-caret" : ""}`} style={{ color: "var(--fg-2)" }}>
          {text}
        </div>
      )}
    </div>
  );
}

// Typewriter for streamed answers. New characters animate in at a steady
// rate (bounded batches every tick), so an answer that arrives in a single
// chunk — langgraph emits the whole response as one assistant_delta — types
// itself out exactly like a token-by-token stream, native parity. The state
// is keyed by message id: polls that rebuild the message list keep the same
// React instance, so the animation never restarts mid-answer.
const TYPO_TICK_MS = 24;
const TYPO_CHARS_PER_TICK = 14;

function AssistantAnswer({
  messageId,
  text,
  streaming,
}: {
  messageId: string;
  text: string;
  streaming?: boolean;
}) {
  const [shown, setShown] = useState(() => (streaming ? 0 : text.length));
  const textRef = useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  const typing = shown < text.length;
  useEffect(() => {
    if (!typing) return;
    const timer = setInterval(() => {
      setShown((cur) => {
        const target = textRef.current.length;
        return cur >= target ? target : Math.min(cur + TYPO_CHARS_PER_TICK, target);
      });
    }, TYPO_TICK_MS);
    return () => clearInterval(timer);
  }, [typing]);

  const caret = typing || !!streaming;
  return (
    <MarkdownText className={`text-[13px] leading-relaxed ${caret ? "stream-caret" : ""}`}>
      {typing ? text.slice(0, shown) : text}
    </MarkdownText>
  );
}

function workspaceLabel(workspace: string): string {
  if (!workspace || workspace === ".") return "";
  const parts = workspace.split(/[/\\]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : workspace;
}

// Frontend-only chat titles (localStorage). The backend has no rename
// endpoint, so a rename overwrites the displayed title on this machine only.
const TITLES_KEY = "operating-agent:chat-titles";

function loadTitleOverrides(): Record<string, string> {
  try {
    const raw = JSON.parse(localStorage.getItem(TITLES_KEY) || "{}");
    return raw && typeof raw === "object" ? (raw as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function saveTitleOverrides(overrides: Record<string, string>) {
  try {
    localStorage.setItem(TITLES_KEY, JSON.stringify(overrides));
  } catch {
    // ignore — storage may be blocked in webview
  }
}

export function ChatWorkspace({
  track,
  onOpenSettings,
  onBackToPicker,
}: {
  track: "native" | "langgraph";
  onOpenSettings: () => void;
  onBackToPicker: () => void;
}) {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const selectedRef = useRef<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [events, setEvents] = useState<EventResponse[]>([]);
  const [permissions, setPermissions] = useState<PermissionResponse[]>([]);
  const [pendingApprovals, setPendingApprovals] = useState<Array<{ id: string; tool_name: string; risk_level: string }>>([]);
  const [composer, setComposer] = useState("");
  const [sending, setSending] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [dialog, setDialog] = useState<DialogState>(null);
  const [pendingTaskId, setPendingTaskId] = useState<string | null>(null);
  // Live langgraph content, keyed by task id. The 2s poll rebuilds messages
  // from persisted tasks (which lack a final message until the run ends), so
  // without these refs each poll would wipe the streaming bubble that SSE
  // just painted — the flicker where the answer appears then vanishes.
  const liveTextRef = useRef(new Map<string, string>());
  const liveThinkRef = useRef(new Map<string, string>());
  const pendingTaskIdRef = useRef<string | null>(null);
  useEffect(() => {
    pendingTaskIdRef.current = pendingTaskId;
  }, [pendingTaskId]);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);
  const [search, setSearch] = useState("");
  const [workspace, setWorkspace] = useState(() => loadSettings().workspace || ".");
  const [titleOverrides, setTitleOverrides] = useState<Record<string, string>>(loadTitleOverrides);
  const listRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const refreshGenerationRef = useRef(0);
  // Ref to current messages for merge logic to avoid duplicating live bubbles
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  // Stable Activity keys for track rows without backend sequences (langgraph):
  // the SSE replay and the REST history describe the same events, so both are
  // keyed by content signature. Without this, every 2s poll remaps keys 1..N
  // while live appends use length+1, churning rows and sticking each row's
  // open-state onto the wrong event.
  const eventKeyRef = useRef(0);
  const eventSigRef = useRef(new Map<string, number>());
  const streamActivityIndexRef = useRef(0);
  const keyForActivity = useCallback((type: string, data: unknown, identity?: string | number): number => {
    const sig = activitySignature(type, data, identity);
    const known = eventSigRef.current.get(sig);
    if (known !== undefined) return known;
    eventKeyRef.current += 1;
    eventSigRef.current.set(sig, eventKeyRef.current);
    return eventKeyRef.current;
  }, []);
  // A new chat gets fresh keys; re-selects and polls keep them stable.
  useEffect(() => {
    eventKeyRef.current = 0;
    eventSigRef.current = new Map();
    streamActivityIndexRef.current = 0;
    // Live buffers belong to one selected conversation. Keeping a completed
    // task's unpersisted text here lets a later refresh merge it into another
    // turn while the database catches up.
    liveTextRef.current.clear();
    liveThinkRef.current.clear();
  }, [selected, track]);

  // Workspace adopted from a task below: saveSettings dispatches synchronously,
  // so this listener would otherwise re-enter with the stale closure workspace
  // and wipe the just-loaded conversation.
  const adoptedWorkspaceRef = useRef<string | null>(null);
  useEffect(() => {
    const onSettings = (event: Event) => {
      const next = (event as CustomEvent<{ workspace?: string }>).detail?.workspace;
      if (typeof next !== "string" || !next.trim()) return;
      if (next === adoptedWorkspaceRef.current) {
        adoptedWorkspaceRef.current = null;
        return;
      }
      if (next !== workspace) {
        setWorkspace(next);
        setSelected(null);
        setMessages([]);
        setEvents([]);
        setPendingApprovals([]);
        setPendingTaskId(null);
      }
    };
    window.addEventListener("operating-agent:settings", onSettings);
    return () => window.removeEventListener("operating-agent:settings", onSettings);
  }, [workspace]);

  const selectWorkspace = (value: string) => {
    const next = value.trim() || ".";
    setWorkspace(next);
    if (next !== workspace) {
      setSelected(null);
      setMessages([]);
      setEvents([]);
      setPendingApprovals([]);
      setPendingTaskId(null);
    }
    saveSettings({ ...loadSettings(), workspace: next });
  };

  // Native file-explorer picker (Tauri shell only; hidden in a browser).
  const canBrowse = isTauri();
  const browseWorkspace = async () => {
    const dir = await pickDirectory(workspace);
    if (dir) selectWorkspace(dir);
  };

  const scrollToEnd = useCallback(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), []);

  // Keep the latest answer in view while it streams (native parity: the live
  // bubble scrolls on every delta). Runs on message/event growth only while a
  // send or a langgraph task is in flight, so reading history stays put.
  const autoScrollActive = sending || pendingTaskId !== null;
  const autoScrollActiveRef = useRef(autoScrollActive);
  useEffect(() => {
    autoScrollActiveRef.current = autoScrollActive;
  }, [autoScrollActive]);
  useEffect(() => {
    if (autoScrollActiveRef.current) scrollToEnd();
  }, [messages, events, scrollToEnd]);

  const patchLiveAssistant = useCallback((taskId: string, patch: Partial<ChatMessage>) => {
    const id = `${taskId}-assistant`;
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
    setTimeout(scrollToEnd, 30);
  }, [scrollToEnd]);

  // ——— load chats ———
  const refreshChats = useCallback(async () => {
    const overrides = loadTitleOverrides();
    setTitleOverrides(overrides);
    if (track === "native") {
      let sessions: SessionResponse[];
      try {
        // Evaluation sessions are intentionally kept in the native transcript
        // store, but older runs may use a different resolved workspace string.
        // Fetch the full session list and retain the selected workspace plus the
        // reserved evaluation namespace so native benchmark chats remain visible.
        sessions = await nativeApi.listSessions({ limit: 100 });
      } catch {
        // Keep the sidebar intact while the API is restarting. The polling
        // effect below retries once the server is reachable again.
        return;
      }
      const items: ChatItem[] = sessions.filter((s) => s.workspace === workspace || s.id.startsWith("evaluation-")).map((s) => ({
        kind: "native" as const,
        id: s.id,
        title: overrides[`native:${s.id}`] || s.title || s.id.slice(0, 12),
        subtitle: workspaceLabel(s.workspace),
        updatedAt: s.updated_at || s.created_at || "",
      }));
      setChats(items);
      const current = selectedRef.current;
      if (!current && items[0]) setSelected(items[0].id);
      else if (current && !items.some((item) => item.id === current)) setSelected(null);
    } else {
      let threads: ThreadResponse[];
      try {
        threads = await taskApi.listThreads({ limit: 100 });
      } catch {
        // Do not turn a temporary API outage into an empty chat history.
        return;
      }
      // Evaluation executions use the reserved ``evaluation-`` thread
      // namespace. They have their own transcript browser in Evaluate and
      // must not leak into a user's normal LangGraph conversation list.
      const items: ChatItem[] = threads
        .filter((thread) => !thread.id.startsWith("evaluation-"))
        .map((t) => ({
        kind: "langgraph" as const,
        id: t.id,
        title: overrides[`langgraph:${t.id}`] || t.title || t.id.slice(0, 12),
        subtitle: `${t.task_count} tasks`,
        updatedAt: t.updated_at,
        }));
      setChats(items);
      const current = selectedRef.current;
      if (!current && items[0]) setSelected(items[0].id);
      else if (current && !items.some((item) => item.id === current)) setSelected(null);
    }
  }, [track, workspace]);

  const refreshConversation = useCallback(
    async (id: string) => {
      const generation = ++refreshGenerationRef.current;
      const isCurrent = () => generation === refreshGenerationRef.current;
      if (track === "native") {
        try {
          const conv = await nativeApi.getConversation(id);
          if (!isCurrent()) return;
          const msgs: ChatMessage[] = conv.messages.map((m) => {
            const parts = m.parts as Array<Record<string, unknown>>;
            const firstText = parts.find((p) =>
              p.text && p.part_type !== "reasoning" && p.part_type !== "compaction" && p.hidden !== true,
            )?.text as string | undefined;
            const thinking = parts.find((p) => p.part_type === "reasoning" && p.text)?.text as string | undefined;
            const toolParts = parts.filter((p) => p.name);
            return {
              id: m.id,
              role: m.role as ChatMessage["role"],
              text: firstText || (m.parts.length ? JSON.stringify(m.parts[0]).slice(0, 200) : ""),
              parts: m.parts,
              model: m.model,
              time: m.created_at || undefined,
              toolCalls: toolParts.map((p) => ({ name: String(p.name), status: String(p.status || "completed"), preview: String(p.output || p.error || "").slice(0, 120) })),
              thinking: thinking || undefined,
            };
          });
          setMessages(msgs);
          const allEvents = await nativeApi.getEvents(id, 0).catch(() => [] as EventResponse[]);
          if (!isCurrent()) return;
          // Activity covers the current response only: keep events from the
          // latest top-level run (helper runs carry "/" in their run id).
          // Token deltas are excluded — they drive the streaming bubble, and
          // one row per chunk would bury the milestones.
          const visible = allEvents.filter((e) => isActivityEvent(e.type));
          let currentRun = "";
          for (let i = visible.length - 1; i >= 0; i--) {
            const rid = visible[i].run_id || "";
            if (!rid || rid.includes("/")) continue;
            currentRun = rid;
            break;
          }
          setEvents(currentRun ? visible.filter((e) => e.run_id === currentRun).slice(-100) : visible.slice(-100));
          const perms = await nativeApi.listPermissions(id).catch(() => [] as PermissionResponse[]);
          if (isCurrent()) setPermissions(perms);
        } catch {
          // A transient refresh failure must not erase the last rendered
          // response. The next refresh will reconcile it from the API.
        }
      } else {
        try {
          // Do not turn a transient task-list failure into an empty thread.
          // Doing so used to remove the live answer and keep polling forever.
          const tasks = await taskApi.listThreadTasks(id, { limit: 100 });
          if (!isCurrent()) return;
          const chosen = loadSettings().workspace;
          if (
            tasks[0]?.workspace &&
            tasks[0].workspace !== workspace &&
            !(chosen && chosen !== ".")
          ) {
            adoptedWorkspaceRef.current = tasks[0].workspace;
            setWorkspace(tasks[0].workspace);
            saveSettings({ ...loadSettings(), workspace: tasks[0].workspace });
          }
          // Persisted reasoning per task → one Thought-process block on each
          // assistant bubble, native parity: thinking survives settle and
          // reload instead of vanishing when the live refs are cleared.
          // reasoning_delta is excluded from Activity rows, so it is
          // collected here; scoping by task id keeps one turn's reasoning
          // from ever leaking into another turn's bubble.
          const ev = await taskApi.listThreadEvents(id).catch(() => null);
          if (!isCurrent()) return;
          const thinkByTask = new Map<string, string>();
          if (ev) {
            for (const e of ev) {
              if (e.type !== "reasoning_delta") continue;
              const text = String((e.payload as Record<string, unknown> | null)?.text || "").trim();
              if (!text) continue;
              const prevT = thinkByTask.get(e.task_id);
              thinkByTask.set(e.task_id, prevT ? `${prevT}\n\n${text}` : text);
            }
          }
          const msgs: ChatMessage[] = tasks
            .slice()
            .reverse()
            .flatMap((t) => [
              { id: `${t.id}-user`, role: "user" as const, text: t.goal, time: t.created_at },
              ...(t.final_message
                ? [{ id: `${t.id}-assistant`, role: "assistant" as const, text: t.final_message, thinking: thinkByTask.get(t.id), time: t.created_at }]
                : t.error
                  ? [{ id: `${t.id}-error`, role: "assistant" as const, text: `Run failed: ${t.error}`, time: t.created_at }]
                  : []),
            ]);
// Re-apply live SSE content the server has not persisted yet. The
          // polling response can lag behind the stream, so rebuilding the
          // list from tasks alone would drop the live bubble on every poll
          // and it would flicker (or vanish entirely until the run settles).
          // Instead, carry the existing live bubble over — the SSE stream
          // keeps it fresh in place — and only replace it once the server
          // has a persisted answer for that task.
          const merged = [...msgs];
          const liveTaskIds = new Set([
            ...liveTextRef.current.keys(),
            ...liveThinkRef.current.keys(),
          ]);
          for (const taskId of liveTaskIds) {
            const hasServerAnswer = merged.some(
              (m) => m.id === `${taskId}-assistant` || m.id === `${taskId}-error`,
            );
            if (hasServerAnswer) {
              // The run settled: the persisted bubble is authoritative.
              liveTextRef.current.delete(taskId);
              liveThinkRef.current.delete(taskId);
              continue;
            }
            const liveText = liveTextRef.current.get(taskId) || "";
            const liveThink = liveThinkRef.current.get(taskId) ?? thinkByTask.get(taskId);
            const existing = messagesRef.current.find((m) => m.id === `${taskId}-assistant`);
            if (!liveText && !liveThink && !existing) continue;
            const entry: ChatMessage = existing
              ? {
                  ...existing,
                  text: liveText || existing.text || "",
                  thinking: liveThink || existing.thinking,
                  streaming: pendingTaskIdRef.current === taskId,
                }
              : {
                  id: `${taskId}-assistant`,
                  role: "assistant",
                  text: liveText,
                  thinking: liveThink || undefined,
                  streaming: pendingTaskIdRef.current === taskId,
                  time: new Date().toISOString(),
                };
            const userIdx = merged.findIndex((m) => m.id === `${taskId}-user`);
            if (userIdx >= 0) merged.splice(userIdx + 1, 0, entry);
            else merged.push(entry);
          }
          setMessages(merged);
          if (pendingTaskIdRef.current && autoScrollActiveRef.current) {
            setTimeout(scrollToEnd, 30);
          }
          setPendingTaskId((cur) => {
            const current = cur ? tasks.find((task) => task.id === cur) : undefined;
            if (current && TERMINAL_TASK_STATUSES.has(current.status || "")) return null;
            if (cur) return cur;
            const active = tasks.find((task) =>
              ACTIVE_TASK_STATUSES.has(task.status || ""),
            );
            return active?.id || null;
          });
          // Activity covers the current response only: keep events from the
          // latest task in this thread. Rows are keyed by content signature so
          // a poll maps the same events to the same keys as the live stream.
          // `ev` was fetched above (it also feeds the Thought-process blocks).
          const latestTaskId = tasks.length > 0 ? tasks[0].id : "";
          if (isCurrent() && ev) {
            const filtered = ev.filter(
              (e) => (!latestTaskId || e.task_id === latestTaskId) && isActivityEvent(e.type),
            );
            const start = Math.max(0, filtered.length - 100);
            setEvents(
              filtered.slice(start).map((e, index) => ({
                sequence: keyForActivity(e.type, e.payload, start + index),
                type: e.type,
                session_id: id,
                run_id: e.task_id,
                data: e.payload,
                time: null,
              })),
            );
          }
          const approvals = await taskApi.listApprovals().catch(() => null);
          if (isCurrent() && approvals) {
            setPendingApprovals(approvals.filter((a) => tasks.some((t) => t.id === a.task_id)));
          }
        } catch {
          // A transient refresh failure must not erase the last rendered
          // response. The next active poll will reconcile it from the API.
        }
      }
    },
    [track, workspace, keyForActivity],
  );

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      await refreshChats();
      if (!stopped) timer = setTimeout(poll, 5000);
    };
    void poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [refreshChats]);

  // Switching chats must never show the previous chat's responses while the
  // new one loads: clear immediately and show a skeleton until its fetch
  // settles. Background polls reuse refreshConversation but must not touch
  // this loading state. A send that just created its session/thread owns its
  // live bubbles, so it suppresses this reload (it refreshes itself at the end).
  const [convLoading, setConvLoading] = useState(false);
  const suppressSelectLoadRef = useRef(false);
  useEffect(() => {
    if (!selected) {
      setMessages([]);
      setEvents([]);
      setPendingTaskId(null);
      setConvLoading(false);
      return;
    }
    if (suppressSelectLoadRef.current) {
      suppressSelectLoadRef.current = false;
      return;
    }
    setConvLoading(true);
    setMessages([]);
    setEvents([]);
    setPendingApprovals([]);
    setPendingTaskId(null);
    let cancelled = false;
    void refreshConversation(selected).finally(() => {
      if (!cancelled) setConvLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selected, refreshConversation]);

  useEffect(() => {
    if (!selected || track !== "langgraph" || !pendingTaskId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      await refreshConversation(selected);
      if (!stopped) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 2000);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [selected, track, pendingTaskId, refreshConversation]);

  // Poll only while a run is active. Refreshing the conversation already
  // fetches approvals for the selected thread, so a second independent poll
  // otherwise doubles the request rate and can make stale approval banners look
  // like repeated approval requests.
  useEffect(() => {
    const t = setInterval(async () => {
      if (track !== "native" || !selected || !sending) return;
      const perms = await nativeApi.listPermissions(selected).catch(() => [] as PermissionResponse[]);
      setPermissions(perms);
    }, 2500);
    return () => clearInterval(t);
  }, [track, selected, sending]);

  // ——— actions ———
  const onNewChat = async () => {
    const title = `Chat ${new Date().toLocaleTimeString()}`;
    if (track === "native") {
      try {
        const s = await nativeApi.createSession({ title, workspace, agent: "build" });
        setChats((prev) => [{ kind: "native", id: s.id, title: s.title || s.id, subtitle: workspaceLabel(s.workspace), updatedAt: s.updated_at || s.created_at || "" }, ...prev]);
        suppressSelectLoadRef.current = true;
        setSelected(s.id);
        setMessages([]);
      } catch (e) {
        setDialog({ kind: "error", message: (e as Error).message });
      }
    } else {
      try {
        const thread = await taskApi.createThread(title);
        setChats((prev) => [{ kind: "langgraph", id: thread.id, title: thread.title || thread.id, subtitle: "0 tasks", updatedAt: thread.updated_at }, ...prev]);
        suppressSelectLoadRef.current = true;
        setSelected(thread.id);
        setMessages([]);
        setEvents([]);
      } catch (e) {
        setDialog({ kind: "error", message: (e as Error).message });
      }
    }
  };

  const applyRename = (title: string) => {
    if (!selected) return;
    // Frontend-only: the backend has no rename endpoint, so the override is
    // kept in localStorage and applied whenever the list is rebuilt.
    setTitleOverrides((prev) => {
      const updated = { ...prev, [`${track}:${selected}`]: title };
      saveTitleOverrides(updated);
      return updated;
    });
    setChats((prev) => prev.map((c) => (c.id === selected ? { ...c, title } : c)));
    setDialog(null);
  };

  const doForkChat = async () => {
    if (track !== "native" || !selected) return;
    setDialog(null);
    try {
      const f = await nativeApi.forkSession(selected, `${selectedMeta?.title || selected} (fork)`);
        setChats((prev) => [{ kind: "native", id: f.id, title: f.title || f.id, subtitle: workspaceLabel(f.workspace), updatedAt: f.updated_at || f.created_at || "" }, ...prev]);
      setSelected(f.id);
    } catch (e) {
      setDialog({ kind: "error", message: (e as Error).message });
    }
  };

  const onDeleteChat = async (id: string, kind: "native" | "langgraph") => {
    setDialog(null);
    try {
      if (kind === "native") {
        await nativeApi.deleteSession(id);
      } else {
        await taskApi.deleteThread(id);
      }
      setChats((prev) => prev.filter((c) => c.id !== id));
      setTitleOverrides((prev) => {
        if (!(`${track}:${id}` in prev)) return prev;
        const updated = { ...prev };
        delete updated[`${track}:${id}`];
        saveTitleOverrides(updated);
        return updated;
      });
      if (selected === id) {
        setSelected(null);
        setMessages([]);
        setEvents([]);
      }
    } catch (e) {
      setDialog({ kind: "error", message: (e as Error).message });
    }
  };

  const patchLiveMessage = (id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
    setTimeout(scrollToEnd, 30);
  };

  const resolveApproval = async (callId: string, allowed: boolean) => {
    try {
      if (track === "native") {
        await nativeApi.resolvePermission(callId, { allowed, duration: "once" });
        setPermissions((prev) => prev.filter((item) => item.call_id !== callId));
      } else {
        await taskApi.resolveApproval(callId, { approved: allowed });
        setPendingApprovals((prev) => prev.filter((item) => item.id !== callId));
      }
    } catch (error) {
      setDialog({ kind: "error", message: (error as Error).message });
    }
  };

  const onSend = async () => {
    const text = composer.trim();
    if (!text || sending) return;
    // A chat is an explicit user-owned container. Do not create a session or
    // thread as a side effect of sending; only New chat provisions one.
    if (!selected) {
      setDialog({
        kind: "error",
        message: "Create a new chat before sending a message.",
      });
      return;
    }
    setSending(true);
    const userMsg: ChatMessage = { id: `local-${Date.now()}`, role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    setEvents([]);
    setComposer("");
    setTimeout(scrollToEnd, 50);

    if (track === "native") {
      const sid = selected;
      // Live assistant bubble: deltas grow this message in place, so the
      // answer streams instead of popping in as one full block at the end.
      const liveId = `live-${Date.now()}`;
      setMessages((prev) => [...prev, { id: liveId, role: "assistant", text: "", thinking: "", streaming: true }]);
      try {
        const url = nativeApi.sendMessageUrl(sid);
        const settings = loadSettings();
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            limits: {
              max_turns: Number(settings.maxTurns) || 10,
              max_cost_usd: Number(settings.maxCost) || 0.05,
            },
          }),
        });
        if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
        let acc = "";
        let thinkAcc = "";
        let finalAnswer = "";
        await readSSEStream(res.body, (frame) => {
          try {
            const payload = JSON.parse(frame.data);
            const eventType = frame.event || payload?.type || "";
            const eventData = payload?.data || payload;
            if (eventType === "assistant_delta") {
              const delta = String(eventData?.text || "");
              if (delta) {
                acc += delta;
                patchLiveMessage(liveId, { text: acc });
              }
            } else if (eventType === "reasoning_delta") {
              const delta = String(eventData?.text || "");
              if (delta) {
                thinkAcc += delta;
                patchLiveMessage(liveId, { thinking: thinkAcc });
              }
            } else if (eventType === "run_receipt") {
              finalAnswer = String(eventData?.final_message || eventData?.final_text || "");
            } else if (eventType === "state") {
              const stateAnswer = assistantTextFromState(eventData as Record<string, unknown>);
              if (stateAnswer) finalAnswer = stateAnswer;
            }
            // Deltas stream the live bubble above; only milestones get rows.
            if (payload?.type && isActivityEvent(payload.type)) {
              setEvents((prev) => [...prev.slice(-99), payload as EventResponse]);
            }
          } catch {
            const rawType = frame.event || "sse";
            if (!isActivityEvent(rawType)) return;
            setEvents((prev) => [...prev.slice(-99), {
              sequence: prev.length + 1,
              type: rawType,
              session_id: sid,
              run_id: "",
              data: { raw: frame.data.slice(0, 200) },
              time: null,
            }]);
          }
        });
        // Settle the live bubble with the receipt's final text when present;
        // it is the same content that streamed, so there is no pop-in.
        patchLiveMessage(liveId, { text: finalAnswer || acc, thinking: thinkAcc || undefined, streaming: false });
        if (sid) refreshConversation(sid);
      } catch (e) {
        patchLiveMessage(liveId, { text: `Error: ${(e as Error).message}`, thinking: undefined, streaming: false });
      } finally {
        setSending(false);
      }
    } else {
      // langgraph
      try {
        // Each task owns exactly one live assistant bubble. Drop any stale
        // stream residue from a prior task before creating the next one.
        liveTextRef.current.clear();
        liveThinkRef.current.clear();
        const body = { goal: text, track: "langgraph" as const, workspace, metadata: {} };
        const task = await taskApi.createThreadTask(selected, body);
        setPendingTaskId(task.id);
        // Live assistant bubble, native parity: deltas grow this message in
        // place, so the answer (and Thought process) streams instead of
        // popping in as one full block at the end.
        liveTextRef.current.set(task.id, "");
        setMessages((prev) => [
          ...prev,
          { id: `${task.id}-assistant`, role: "assistant", text: "", thinking: "", streaming: true, time: new Date().toISOString() },
        ]);
        setTimeout(scrollToEnd, 50);
        // stream via SSE for a bit
        const url = taskApi.streamEventsUrl(task.thread_id, task.id);
        streamActivityIndexRef.current = 0;
        let closeStream = () => {};
        closeStream = sseSubscribe(
          url,
          (event) => {
            // The stream belongs to the thread it was opened for. If the user
            // switched chats mid-send, never write this run's deltas or
            // receipt into the newly selected conversation — that paints the
            // previous response into the wrong thread.
            if (selectedRef.current !== task.thread_id) return;
            let data: Record<string, unknown> = { raw: event.data.slice(0, 200) };
             try {
              const parsed = JSON.parse(event.data);
              if (parsed && typeof parsed === "object") data = parsed as Record<string, unknown>;
            } catch {
              // Keep raw data for non-JSON events.
            }
            // Same content signature as the refresh path: replayed history and
            // the 2s poll map to identical keys instead of duplicating rows.
            if (isActivityEvent(event.event)) {
              const identity = event.id || (
                typeof data.sequence === "number" || typeof data.sequence === "string"
                  ? data.sequence
                  : streamActivityIndexRef.current
              );
              streamActivityIndexRef.current += 1;
              const key = keyForActivity(event.event, data, identity);
              setEvents((prev) =>
                prev.some((p) => p.sequence === key)
                  ? prev
                  : [...prev.slice(-99), {
                    sequence: key,
                    type: event.event,
                    session_id: task.thread_id,
                    run_id: task.id,
                    data,
                    time: null,
                  }],
              );
              if (autoScrollActiveRef.current) setTimeout(scrollToEnd, 30);
            }
            // Single-writer rule for the answer bubble: only `assistant_delta`
            // fragments (new streamed content) and the `finished` receipt
            // (authoritative full answer) may write it. `state` snapshots are
            // deliberately excluded — they carry the whole checkpoint history,
            // so painting them replays the previous turn's answer and raw step
            // transcripts into the current bubble while the agent works. State
            // still feeds the Activity timeline via isActivityEvent above.
            if (event.event === "reasoning_delta") {
              // Each reasoning_delta event carries one complete paragraph
              // (plan rationale, one verification verdict), so events join
              // with a blank line — identically to the refresh path below,
              // so settling never reflows the text. (assistant_delta stays a
              // plain append: the responder emits the answer as one event.)
              const delta = String(data.text || "").trim();
              if (delta) {
                const prevT = liveThinkRef.current.get(task.id);
                const next = prevT ? `${prevT}\n\n${delta}` : delta;
                liveThinkRef.current.set(task.id, next);
                patchLiveAssistant(task.id, { thinking: next });
              }
            } else if (event.event === "assistant_delta") {
              const delta = String(data.text || "");
              if (delta) {
                const next = (liveTextRef.current.get(task.id) || "") + delta;
                liveTextRef.current.set(task.id, next);
                patchLiveAssistant(task.id, { text: next });
              }
            } else if (event.event === "finished" || event.event === "error") {
              const finalText = String(data.final_message || data.output || "");
              if (finalText) {
                liveTextRef.current.set(task.id, finalText);
                patchLiveAssistant(task.id, {
                  text: finalText,
                  thinking: liveThinkRef.current.get(task.id) || undefined,
                  streaming: false,
                });
              } else {
                liveTextRef.current.delete(task.id);
                liveThinkRef.current.delete(task.id);
                setMessages((prev) => prev.filter((m) => m.id !== `${task.id}-assistant`));
              }
              setPendingTaskId(null);
              closeStream();
            }
          },
          () => closeStream(),
        );
        setTimeout(closeStream, 30000);
        await refreshConversation(task.thread_id);
        await refreshChats();
      } catch (e) {
        setMessages((prev) => [...prev, { id: `err-${Date.now()}`, role: "assistant", text: `Error: ${(e as Error).message}` }]);
      } finally {
        setSending(false);
      }
    }
  };

  const filtered = chats.filter((c) => !search || c.title.toLowerCase().includes(search.toLowerCase()) || c.id.toLowerCase().includes(search.toLowerCase()));
  const selectedMeta = chats.find((c) => c.id === selected);

  return (
    <div className="flex flex-1 min-h-0">
      {/* Sidebar */}
      <div className="w-[300px] shrink-0 flex flex-col border-r" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
        <div className="p-3 pb-2 space-y-2.5">
          <button onClick={onNewChat} className="btn-grad w-full h-9 rounded-xl text-[12px] font-semibold flex items-center justify-center gap-2" style={{ color: "white", border: "1px solid transparent" }}>
            <span className="text-[14px] leading-none">＋</span> New chat
          </button>
          <div className="relative">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[11px]" style={{ color: "var(--fg-3)" }}>⌕</span>
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search chats…" className="field !h-8 !text-[11px] !pl-7" />
          </div>
          {/* Segmented view switch */}
          <div className="grid grid-cols-2 gap-1 p-1 rounded-xl" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
            {(["chats", "analytics"] as const).map((view) => {
              const active = showAnalytics === (view === "analytics");
              return (
                <button
                  key={view}
                  onClick={() => setShowAnalytics(view === "analytics")}
                  className="h-7 rounded-lg text-[11px] font-semibold capitalize"
                  style={{
                    background: active ? "var(--accent-grad)" : "transparent",
                    color: active ? "white" : "var(--fg-2)",
                    boxShadow: active ? "var(--accent-glow)" : "none",
                    transition: "all var(--dur-2) var(--ease-out)",
                  }}
                >
                  {view}
                </button>
              );
            })}
          </div>
          <label className="block">
            <span className="block text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--fg-3)" }}>Working directory</span>
<span className="flex gap-1.5">
              {canBrowse ? (
                <button type="button" onClick={browseWorkspace} className="field !h-8 !text-[11px] mono flex-1 min-w-0 text-left truncate" title="Choose working directory">
                  {workspace === "." ? "Choose workspace folder" : folderName(workspace)}
                </button>
              ) : (
                <input
                  value={workspace}
                  onChange={(e) => setWorkspace(e.target.value)}
                  onBlur={() => selectWorkspace(workspace)}
                  onKeyDown={(e) => e.key === "Enter" && selectWorkspace(workspace)}
                  placeholder="Working directory"
                  className="field mono !h-8 !text-[11px]"
                />
              )}
              {canBrowse && (
                <button
                  onClick={browseWorkspace}
                  title="Choose folder in file explorer"
                  className="hidden"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
                >
                  Browse…
                </button>
              )}
            </span>
          </label>
        </div>

        <div ref={listRef} className="flex-1 overflow-auto px-2 pb-2">
          {showAnalytics ? (
            <AnalyticsView track={track} events={events} live={sending} />
          ) : (
            <>
              <div className="flex items-center px-1.5 pt-1 pb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--fg-3)" }}>Chats</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-md" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>{filtered.length}</span>
                <button
                  onClick={() => {
                    const current = chats.find((c) => c.id === selected);
                    if (current) setDialog({ kind: "rename", current: current.title });
                  }}
                  disabled={!selected}
                  title="Rename selected chat"
                  className="btn-quiet ml-auto h-6 px-2 rounded-md text-[10px] font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
                >
                  Rename
                </button>
              </div>
              <div className="space-y-1">
              {filtered.map((c) => (
                <div
                  key={c.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelected(c.id)}
                  onKeyDown={(e) => e.key === "Enter" && setSelected(c.id)}
                  className="group w-full text-left p-2.5 rounded-xl flex gap-2.5 btn-quiet cursor-pointer"
                  style={{ background: selected === c.id ? "var(--bg-2)" : "transparent", border: `1px solid ${selected === c.id ? "var(--accent-ring)" : "transparent"}`, boxShadow: selected === c.id ? "0 0 0 1px var(--accent-ring), 0 2px 12px rgba(34,211,238,0.10)" : "none" }}
                >
                  <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>{c.kind === "native" ? "◈" : "⬢"}</span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline gap-2">
                      <span className="block text-[12px] font-medium truncate" style={{ color: "var(--fg-0)" }}>{c.title}</span>
                      <span className="ml-auto text-[10px] shrink-0" style={{ color: "var(--fg-3)" }}>{timeAgo(c.updatedAt)}</span>
                    </span>
                    {c.subtitle ? (
                      <span className="block text-[11px] truncate" style={{ color: "var(--fg-2)" }}>{c.subtitle}</span>
                    ) : null}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setDialog({ kind: "delete", id: c.id, chatKind: c.kind, title: c.title }); }}
                    title="Delete chat"
                    className="w-6 h-6 rounded-md grid place-items-center text-[12px] shrink-0 self-center opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
                    style={{ background: "var(--bg-3)", border: "1px solid var(--bg-4)", color: "var(--fg-2)", transition: "opacity var(--dur-1) ease, color var(--dur-1) ease" }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "var(--danger)")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-2)")}
                  >
                    ×
                  </button>
                </div>
              ))}
              {filtered.length === 0 && <div className="text-[11px] p-3" style={{ color: "var(--fg-3)" }}>{chats.length === 0 ? "No chats — create one above." : "No matches."}</div>}
              </div>
            </>
          )}
        </div>

        <div className="p-2.5 border-t space-y-2" style={{ borderColor: "var(--bg-4)" }}>
          {track === "native" && selected && (
            <button
              onClick={() => setDialog({ kind: "fork" })}
              className="btn-quiet w-full h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⑂ Fork chat
            </button>
          )}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={onBackToPicker}
              className="btn-quiet h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⇄ Switch track
            </button>
            <button
              onClick={onOpenSettings}
              className="btn-quiet h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⚙ Settings
            </button>
          </div>
        </div>
      </div>

      {/* Main chat */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* header */}
        <div className="min-h-11 shrink-0 flex items-center gap-3 px-4 py-2 border-b" style={{ borderColor: "var(--bg-4)", background: "var(--bg-0)" }}>
          <span className="w-8 h-8 rounded-xl grid place-items-center text-[13px] font-bold shrink-0" style={{ background: "var(--accent-grad-soft)", border: "1px solid var(--accent-ring)", color: "var(--accent)" }}>{track === "native" ? "◈" : "⬢"}</span>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold truncate font-display">{selectedMeta?.title || "New chat"}</div>
            <div className="text-[11px] truncate" style={{ color: "var(--fg-2)" }}>{selected ? `${selectedMeta?.subtitle || ""}${messages.length ? ` · ${messages.length} messages` : ""}` : "No chat selected — send a message to create one."}</div>
          </div>
          <div className="ml-auto flex gap-1.5">
            <span className="hidden sm:inline text-[10px] font-medium px-2 py-1 rounded-full capitalize" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
              {track}
            </span>
          </div>
        </div>

        {/* permissions strip */}
        {(track === "native" ? permissions.length > 0 : pendingApprovals.length > 0) && (
          <div className="px-3 py-2 border-b flex gap-2 items-center overflow-auto anim-slide-down" style={{ borderColor: "var(--bg-4)", background: "var(--warning-soft)" }}>
            <span className="text-[10px] font-semibold uppercase tracking-wider shrink-0" style={{ color: "var(--warning)" }}>Approval needed</span>
            {(track === "native" ? permissions : pendingApprovals.map((a) => ({ call_id: a.id, tool: a.tool_name, preview: a.risk_level, reason: a.id } as unknown as PermissionResponse))).slice(0, 3).map((p) => (
              <div key={p.call_id} className="flex items-center gap-2 px-2.5 py-1.5 rounded-full text-[11px] font-mono shrink-0" style={{ background: "var(--bg-1)", border: "1px solid var(--warning-soft)", color: "var(--fg-1)" }}>
                <span className="anim-pulse-dot" style={{ color: "var(--warning)" }}>⚠</span> {p.tool} · {p.call_id.slice(0, 6)}
                <button onClick={() => resolveApproval(p.call_id, true)} className="btn-grad ml-1 px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>Allow</button>
                <button onClick={() => resolveApproval(p.call_id, false)} className="btn-quiet px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>Deny</button>
              </div>
            ))}
          </div>
        )}

        {/* messages */}
        <div className="flex-1 overflow-auto p-4 sm:p-6">
          <div className="mx-auto w-full max-w-[760px] space-y-4">
            {convLoading ? (
              <div className="space-y-3 anim-fade-in" aria-label="Loading conversation">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="flex gap-3">
                    <span className="w-7 h-7 rounded-lg skeleton shrink-0" />
                    <div className="rounded-2xl px-3.5 py-2.5 skeleton" style={{ width: `${[72, 58, 64][i]}%`, height: 64 }} />
                  </div>
                ))}
              </div>
            ) : messages.length === 0 && !sending ? (
              <div className="rounded-2xl p-8 text-center hero-glow anim-fade-up" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
                <div className="w-11 h-11 rounded-2xl mx-auto grid place-items-center text-[18px] font-bold" style={{ background: "var(--accent-grad)", color: "#fff", boxShadow: "var(--accent-glow)" }}>{track === "native" ? "◈" : "⬢"}</div>
                <div className="mt-3 text-[15px] font-semibold font-display">Start a <span className="grad-text">new chat</span></div>
                <div className="mt-1 text-[12px] leading-relaxed" style={{ color: "var(--fg-2)" }}>
                  {track === "native" ? "Describe a goal — the agent plans, uses tools, and streams the answer back." : "Describe a goal — it runs as a task and streams progress back."}
                </div>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {["Refactor auth middleware", "Add session forking", "Run tests and fix failures"].map((s) => (
                    <button key={s} onClick={() => setComposer(s)} className="btn-quiet px-3 py-1.5 rounded-full text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}>{s}</button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map((m) => (
                  <div key={m.id} className={`flex gap-3 ${m.streaming ? "" : "anim-fade-up"} ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    {m.role !== "user" && <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0 mt-0.5" style={{ background: "var(--accent-grad)", color: "white", boxShadow: "0 2px 10px rgba(34,211,238,0.2)" }}>◈</span>}
                    <div className={`max-w-[78%] rounded-2xl px-3.5 py-2.5 ${m.role === "user" ? "rounded-br-sm" : "rounded-bl-sm"}`} style={{ background: m.role === "user" ? "var(--accent-grad)" : "var(--bg-1)", color: m.role === "user" ? "white" : "var(--fg-0)", border: `1px solid ${m.role === "user" ? "transparent" : m.streaming ? "var(--accent-ring)" : "var(--bg-4)"}`, boxShadow: m.role === "user" ? "0 2px 14px rgba(34,211,238,0.2)" : "none" }}>
                      {m.role !== "user" && (m.thinking || m.streaming) && (
                        <ThinkingBlock text={m.thinking || ""} live={!!m.streaming} />
                      )}
                      {m.text ? (
                        m.role === "assistant" ? (
                          <AssistantAnswer key={m.id} messageId={m.id} text={m.text} streaming={m.streaming} />
                        ) : (
                          <div className={`text-[13px] leading-relaxed whitespace-pre-wrap break-words ${m.streaming ? "stream-caret" : ""}`}>{m.text}</div>
                        )
                      ) : m.streaming ? (
                        <div className="text-[13px] leading-relaxed stream-caret" style={{ color: "var(--fg-2)" }}> </div>
                      ) : null}
                      {m.toolCalls && m.toolCalls.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {m.toolCalls.map((t, i) => (
                            <div key={i} className="text-[11px] font-mono px-2 py-1 rounded-lg" style={{ background: m.role === "user" ? "rgba(255,255,255,0.15)" : "var(--bg-2)", border: "1px solid var(--bg-4)", color: m.role === "user" ? "white" : "var(--fg-2)" }}>{t.name} · {t.status} {t.preview ? `— ${t.preview}` : ""}</div>
                          ))}
                        </div>
                      )}
                      {m.time && !m.streaming && <div className="mt-1 text-[10px] font-mono" style={{ color: m.role === "user" ? "rgba(255,255,255,0.7)" : "var(--fg-3)" }}>{formatLocalTime(m.time)}</div>}
                    </div>
                    {m.role === "user" && <span className="w-7 h-7 rounded-full grid place-items-center text-[11px] font-semibold shrink-0 mt-0.5" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>you</span>}
                  </div>
                ))}
                {pendingTaskId && !messages.some(m => m.id === `${pendingTaskId}-assistant`) && (
                  <div className="flex gap-3 anim-fade-up">
                    <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0" style={{ background: "var(--accent-grad)", color: "white", boxShadow: "0 2px 10px rgba(34,211,238,0.2)" }}>⬢</span>
                    <div className="max-w-[78%] rounded-2xl rounded-bl-sm px-3.5 py-2.5" style={{ background: "var(--bg-1)", border: "1px solid var(--accent-ring)" }}>
                      <div className="text-[13px] leading-relaxed flex items-center gap-2" style={{ color: "var(--fg-2)" }}>
                        <span className="w-1.5 h-1.5 rounded-full anim-pulse-dot" style={{ background: "var(--accent)" }} />
                        Working on your task…
                      </div>
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </>
            )}
          </div>
        </div>

        {/* composer — floating card, no attached bar */}
        <div className="px-4 sm:px-6 pt-1 pb-4">
          <div
            className="mx-auto max-w-[760px] rounded-2xl p-2 pl-4 flex gap-2 items-end anim-fade-up"
            style={{
              background: "var(--bg-1)",
              border: "1px solid var(--bg-4)",
              boxShadow: "0 12px 32px rgba(2, 8, 20, 0.55), 0 0 0 1px rgba(6, 182, 212, 0.06)",
            }}
          >
            <textarea
              value={composer}
              onChange={(e) => setComposer(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
              placeholder={selected ? "Message the agent…" : "Create a new chat to start messaging…"}
              rows={composer.includes("\n") ? 3 : 1}
              className="flex-1 min-h-[44px] max-h-28 py-2.5 text-[13px] leading-relaxed outline-none resize-none bg-transparent"
              style={{ color: "var(--fg-0)" }}
            />
            <button onClick={onSend} disabled={sending || !selected || !composer.trim()} className="btn-grad h-10 px-5 rounded-xl text-[13px] font-semibold shrink-0" style={{ color: "white", border: "1px solid transparent" }}>
              {sending ? "…" : "Send →"}
            </button>
          </div>
        </div>
      </div>
      {dialog?.kind === "rename" && (
        <PromptDialog
          title="Rename chat"
          subtitle="Stored on this machine and applied whenever the list reloads."
          initialValue={dialog.current}
          placeholder="Chat name"
          confirmLabel="Rename"
          onSubmit={applyRename}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "delete" && (
        <ConfirmDialog
          title="Delete chat?"
          message={`"${dialog.title}" and its full history will be permanently removed.`}
          confirmLabel="Delete"
          danger
          onConfirm={() => onDeleteChat(dialog.id, dialog.chatKind)}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "fork" && (
        <ConfirmDialog
          title="Fork this chat?"
          message="The full conversation is copied into a new chat with its own history. The original stays untouched."
          confirmLabel="Fork chat"
          onConfirm={doForkChat}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "error" && (
        <AlertDialog message={dialog.message} onClose={() => setDialog(null)} />
      )}
    </div>
  );
}
