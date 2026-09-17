import { useCallback, useEffect, useRef, useState } from "react";
import { sseSubscribe, taskApi } from "../../lib/api";
import type { ApprovalResponse, HealthResponse, TaskResponse, ThreadEventResponse, ThreadResponse } from "../../lib/types";
import { formatLocalDateTime } from "../../lib/time";
import { loadSettings, saveSettings } from "../SettingsModal";
import { Card, Label } from "../layout/Shell";
import { folderName, isTauri, pickDirectory } from "../../lib/pickFolder";

function useHealth() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    taskApi.health().then(setData).catch((e: Error) => setErr(e.message));
  }, []);
  return { data, err };
}

export function LangGraphWorkspace() {
  const health = useHealth();
  const [threads, setThreads] = useState<ThreadResponse[]>([]);
  const [threadsErr, setThreadsErr] = useState<string | null>(null);
  const [selectedThread, setSelectedThread] = useState<string | null>(null);
  const [tasks, setTasks] = useState<TaskResponse[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [threadEvents, setThreadEvents] = useState<ThreadEventResponse[]>([]);
  const [approvals, setApprovals] = useState<ApprovalResponse[]>([]);
  const [goal, setGoal] = useState("");
  const [track, setTrack] = useState<"native" | "langgraph">("langgraph");
  const [workspace, setWorkspace] = useState(() => loadSettings().workspace || ".");
  // Workspace this component adopted itself: saveSettings dispatches
  // synchronously, so the listener below would otherwise re-enter with the
  // stale closure workspace and wipe the just-loaded selection.
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
        setSelectedThread(null);
        setSelectedTask(null);
        setTasks([]);
        setThreadEvents([]);
      }
    };
    window.addEventListener("operating-agent:settings", onSettings);
    return () => window.removeEventListener("operating-agent:settings", onSettings);
  }, [workspace]);
  const [streamLog, setStreamLog] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<"tasks" | "events" | "approvals" | "stream">("tasks");
  const browseWorkspace = async () => {
    const dir = await pickDirectory(workspace);
    if (!dir) return;
    setWorkspace(dir);
    saveSettings({ ...loadSettings(), workspace: dir });
    setSelectedThread(null);
    setSelectedTask(null);
    setTasks([]);
    setThreadEvents([]);
  };

  const refreshThreads = useCallback(async () => {
    try {
      const list = await taskApi.listThreads({ limit: 100 });
      setThreads(list);
      setThreadsErr(null);
      if (!selectedThread && list[0]) setSelectedThread(list[0].id);
    } catch (e) {
      setThreadsErr((e as Error).message);
    }
  }, [selectedThread]);

  const refreshTasks = useCallback(async (tid: string) => {
    try {
      const list = await taskApi.listThreadTasks(tid, { limit: 100 });
      setTasks(list);
      const chosen = loadSettings().workspace;
      if (
        list[0]?.workspace &&
        list[0].workspace !== workspace &&
        !(chosen && chosen !== ".")
      ) {
        adoptedWorkspaceRef.current = list[0].workspace;
        setWorkspace(list[0].workspace);
        saveSettings({ ...loadSettings(), workspace: list[0].workspace });
      }
      if (!selectedTask && list[0]) setSelectedTask(list[0].id);
      const ev = await taskApi.listThreadEvents(tid).catch(() => [] as ThreadEventResponse[]);
      setThreadEvents(ev);
    } catch {
      // ignore
    }
  }, [selectedTask, workspace]);

  useEffect(() => {
    refreshThreads();
  }, [refreshThreads]);

  useEffect(() => {
    if (selectedThread) refreshTasks(selectedThread);
  }, [selectedThread, refreshTasks]);

  useEffect(() => {
    if (!selectedThread || !tasks.some((task) => ["created", "pending", "running"].includes(task.status || ""))) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      await refreshTasks(selectedThread);
      if (!stopped) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 3000);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [selectedThread, tasks, refreshTasks]);

  useEffect(() => {
    if (!selectedThread || (!tasks.some((task) => ["created", "pending", "running"].includes(task.status || "")) && approvals.length === 0)) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const next = await taskApi.listApprovals().catch(() => null);
      if (!stopped && next) setApprovals(next.filter((approval) => tasks.some((task) => task.id === approval.task_id)));
      if (!stopped) timer = setTimeout(poll, 2500);
    };
    timer = setTimeout(poll, 2500);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [selectedThread, tasks, approvals.length]);

  const onCreateTask = async (threadId?: string) => {
    if (!goal.trim()) return;
    try {
      const body = { goal: goal.trim(), track: track as "native" | "langgraph", workspace: workspace || undefined, metadata: {} };
      const t = threadId ? await taskApi.createThreadTask(threadId, body) : await taskApi.createTask(body);
      setGoal("");
      setStreamLog((prev) => [...prev, `POST /tasks → ${t.id} thread=${t.thread_id} track=${t.track} status=${t.status}`]);
      if (threadId) refreshTasks(threadId);
      else refreshThreads();
      setSelectedTask(t.id);
      if (!threadId) setSelectedThread(t.thread_id);
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onResume = async () => {
    if (!selectedTask) return;
    const resumeValue = prompt("resume_value (JSON or text) — leave empty for none");
    let parsed: unknown = resumeValue || null;
    if (resumeValue) {
      try {
        parsed = JSON.parse(resumeValue);
      } catch {
        parsed = resumeValue;
      }
    }
    try {
      // try thread-scoped first if we have a thread
      if (!selectedThread) return;
      const t = await taskApi.resumeTask(selectedThread, selectedTask, { resume_value: parsed });
      setStreamLog((prev) => [...prev, `POST resume → ${t.id} status=${t.status}`]);
      if (selectedThread) refreshTasks(selectedThread);
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onResolve = async (id: string, approved: boolean) => {
    const note = prompt("note (optional)") || undefined;
    try {
      await taskApi.resolveApproval(id, { approved, note });
      setApprovals((prev) => prev.filter((a) => a.id !== id));
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const openSSE = () => {
    if (!selectedTask || !selectedThread) return;
    const url = taskApi.streamEventsUrl(selectedThread, selectedTask);
    let closeStream = () => {};
    closeStream = sseSubscribe(url, (event) => {
      setStreamLog((prev) => [...prev.slice(-200), `${event.event}: ${event.data.slice(0, 300)}`]);
      if (event.event === "finished" || event.event === "error") closeStream();
    }, () => {
      setStreamLog((prev) => [...prev, "sse error/closed"]);
      closeStream();
    });
    setStreamLog((prev) => [...prev, `GET ${url} — SSE open`]);
    setTimeout(closeStream, 30000);
  };

  const openWS = () => {
    if (!selectedTask || !selectedThread) return;
    const url = taskApi.wsStreamUrl(selectedThread, selectedTask);
    setStreamLog((prev) => [...prev, `WS ${url} — opening`]);
    try {
      const ws = new WebSocket(url);
      ws.onmessage = (e) => setStreamLog((prev) => [...prev.slice(-200), `ws: ${String(e.data).slice(0, 300)}`]);
      ws.onerror = () => setStreamLog((prev) => [...prev, "ws error"]);
      ws.onclose = () => setStreamLog((prev) => [...prev, "ws closed"]);
      setTimeout(() => ws.close(), 30000);
    } catch (e) {
      setStreamLog((prev) => [...prev, `ws failed: ${(e as Error).message}`]);
    }
  };

  return (
    <div className="flex flex-1 min-h-0">
      {/* threads */}
      <div className="w-[320px] shrink-0 border-r flex flex-col" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
        <div className="p-3 border-b space-y-3" style={{ borderColor: "var(--bg-4)" }}>
          <div className="flex items-center gap-2">
            <Label>Threads</Label>
            <span className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>{threads.length}</span>
          </div>
          {threadsErr && <div className="text-[11px] font-mono p-2 rounded" style={{ background: "var(--danger-soft)", color: "var(--danger)", border: "1px solid var(--danger-soft)" }}>{threadsErr}</div>}
          <div className="text-[11px] font-mono p-2 rounded" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
            <div>GET /health</div>
            <div style={{ color: "var(--fg-1)" }}>{health.data ? `${health.data.status} · ${health.data.repository} · ${health.data.tracks.join(", ")}` : health.err || "…"}</div>
          </div>
          <div className="grid gap-2">
            <textarea value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Goal — POST /tasks {goal, track, workspace}" rows={3} className="rounded-xl px-3 py-2 text-[12px] outline-none resize-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-0)" }} />
            <div className="flex gap-2">
              <select value={track} onChange={(e) => setTrack(e.target.value as "native" | "langgraph")} className="h-8 px-2 rounded-lg text-[11px] outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>
                <option value="langgraph">langgraph</option>
                <option value="native">native</option>
              </select>
              <button type="button" onClick={browseWorkspace} disabled={!isTauri()} className="flex-1 h-8 px-2 rounded-lg text-left text-[11px] font-mono truncate disabled:opacity-60" title="Choose task workspace folder" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>
                {workspace === "." ? "Choose workspace folder" : folderName(workspace)}
              </button>
            </div>
            <div className="flex gap-2">
              <button onClick={() => onCreateTask()} className="btn-grad flex-1 h-8 rounded-lg text-[11px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>POST /tasks</button>
              <button onClick={() => selectedThread && onCreateTask(selectedThread)} disabled={!selectedThread} className="flex-1 h-8 rounded-lg text-[11px] font-medium disabled:opacity-50" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>POST /threads/{"{id}"}/tasks</button>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-2 space-y-1">
          {threads.map((th) => (
            <button key={th.id} onClick={() => setSelectedThread(th.id)} className="w-full text-left p-2.5 rounded-xl" style={{ background: selectedThread === th.id ? "var(--bg-2)" : "transparent", border: `1px solid ${selectedThread === th.id ? "var(--accent-ring)" : "transparent"}` }}>
              <div className="text-[12px] font-medium truncate">{th.title || th.id}</div>
              <div className="text-[11px] font-mono truncate" style={{ color: "var(--fg-2)" }}>{th.id.slice(0, 10)} · {th.task_count} tasks</div>
              <div className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>{formatLocalDateTime(th.updated_at)}</div>
            </button>
          ))}
          {threads.length === 0 && <div className="text-[11px] p-3" style={{ color: "var(--fg-3)" }}>No threads — POST /tasks to create.</div>}
        </div>
      </div>

      {/* tasks + detail */}
      <div className="flex-1 flex flex-col min-w-0">
        {!selectedThread ? (
          <div className="flex-1 grid place-items-center p-8 text-center">
            <div className="rounded-2xl p-6" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}><div className="text-[13px] font-medium">Select a thread</div><div className="text-[12px] mt-1" style={{ color: "var(--fg-2)" }}>Then drive tasks, events, and approvals.</div></div>
          </div>
        ) : (
          <>
            <div className="px-4 py-3 border-b flex flex-wrap gap-2" style={{ borderColor: "var(--bg-4)" }}>
              <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>{selectedThread} · {tasks.length} tasks · <span style={{ color: "var(--fg-1)" }}>{tasks.find((t) => t.id === selectedTask)?.goal?.slice(0, 60) || "—"}</span></div>
              <div className="ml-auto flex gap-1.5">
                <button onClick={onResume} disabled={!selectedTask} className="h-7 px-2.5 rounded-lg text-[11px] font-medium disabled:opacity-50" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>POST resume</button>
                <button onClick={openSSE} disabled={!selectedTask || !selectedThread} className="h-7 px-2.5 rounded-lg text-[11px] font-medium disabled:opacity-50" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>SSE /threads/{"{thread}"}/tasks/{"{id}"}/events</button>
                <button onClick={openWS} disabled={!selectedTask || !selectedThread} className="h-7 px-2.5 rounded-lg text-[11px] font-medium disabled:opacity-50" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>WS /ws/threads/{"{thread}"}/tasks/{"{id}"}</button>
              </div>
            </div>

            <div className="flex gap-1 px-3 py-2 border-b" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
              {(["tasks", "events", "approvals", "stream"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className="h-7 px-3 rounded-full text-[11px] font-medium capitalize"
                  style={{ background: activeTab === t ? "var(--accent-grad)" : "var(--bg-2)", color: activeTab === t ? "white" : "var(--fg-2)", border: "1px solid var(--bg-4)", boxShadow: activeTab === t ? "var(--accent-glow)" : "none" }}
                >
                  {t} {t === "approvals" && approvals.length ? `(${approvals.length})` : t === "tasks" ? `(${tasks.length})` : ""}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-auto p-4 space-y-3">
              {activeTab === "tasks" && (
                <div className="space-y-2">
                  <Label>GET /threads/{"{thread_id}"}/tasks — GET /threads/{"{id}"}/tasks/{"{task_id}"}</Label>
                  {tasks.map((t) => (
                    <button key={t.id} onClick={() => setSelectedTask(t.id)} className="w-full text-left rounded-xl p-3" style={{ background: selectedTask === t.id ? "var(--bg-2)" : "var(--bg-1)", border: `1px solid ${selectedTask === t.id ? "var(--accent-ring)" : "var(--bg-4)"}` }}>
                      <div className="flex gap-2 items-start">
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{ background: t.status === "completed" ? "var(--success-soft)" : "var(--bg-3)", border: "1px solid var(--bg-4)", color: t.status === "completed" ? "var(--success)" : "var(--fg-2)" }}>{t.status || "pending"} · {t.track}</span>
                        <span className="ml-auto text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>{t.id.slice(0, 8)} · {t.run_id?.slice(0, 8) || "no run"}</span>
                      </div>
                      <div className="text-[12px] font-medium mt-1">{t.goal}</div>
                      <div className="text-[11px] font-mono truncate" style={{ color: "var(--fg-2)" }}>{t.final_message || t.error || "—"} {t.trace_id ? `· trace ${t.trace_id.slice(0, 8)}` : ""}</div>
                      <div className="text-[10px] font-mono mt-1" style={{ color: "var(--fg-3)" }}>{formatLocalDateTime(t.created_at)} · {t.workspace ? folderName(t.workspace) : ""}</div>
                    </button>
                  ))}
                  {tasks.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No tasks in this thread — POST /threads/{selectedThread}/tasks</div>}
                </div>
              )}

              {activeTab === "events" && (
                <div className="space-y-2">
                  <Label>GET /threads/{"{thread_id}"}/events</Label>
                  <Card className="p-3 max-h-[520px] overflow-auto">
                    <div className="space-y-1">
                      {threadEvents.map((e, i) => (
                        <div key={`${e.task_id}-${i}`} className="flex gap-2 text-[11px] font-mono p-1.5 rounded" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                          <span style={{ color: "var(--fg-2)" }}>{e.task_id.slice(0, 6)}</span>
                          <span style={{ color: "var(--accent)" }}>{e.type}</span>
                          <span className="truncate" style={{ color: "var(--fg-1)" }}>{JSON.stringify(e.payload).slice(0, 180)}</span>
                        </div>
                      ))}
                      {threadEvents.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No events.</div>}
                    </div>
                  </Card>
                </div>
              )}

              {activeTab === "approvals" && (
                <div className="space-y-2">
                  <Label>GET /approvals · POST /approvals/{"{id}"}/resolve</Label>
                  {approvals.map((a) => (
                    <Card key={a.id} className="p-3 space-y-2">
                      <div className="text-[12px] font-mono font-medium">{a.tool_name} · {a.id.slice(0, 8)} · {a.risk_level}</div>
                      <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>task {a.task_id.slice(0, 8)} · {JSON.stringify(a.arguments).slice(0, 200)}</div>
                      <div className="flex gap-2">
                        <button onClick={() => onResolve(a.id, true)} className="btn-grad h-7 px-3 rounded-lg text-[11px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>Approve</button>
                        <button onClick={() => onResolve(a.id, false)} className="h-7 px-3 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--danger)" }}>Deny</button>
                      </div>
                    </Card>
                  ))}
                  {approvals.length === 0 && <div className="text-[11px] p-3 rounded-xl" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-3)" }}>No pending approvals — GET /approvals returns []</div>}
                </div>
              )}

              {activeTab === "stream" && (
                <div className="space-y-2">
                  <Label>Stream log (SSE + WS)</Label>
                  <Card className="p-3">
                    <pre className="text-[11px] font-mono whitespace-pre-wrap break-words max-h-[520px] overflow-auto" style={{ color: "var(--fg-1)" }}>{streamLog.length ? streamLog.join("\n") : "No stream yet — open SSE or WS for the selected task."}</pre>
                  </Card>
                  <div className="text-[11px] font-mono" style={{ color: "var(--fg-3)" }}>GET /threads/{"{thread}"}/tasks/{"{id}"}/events (SSE) · WS /ws/threads/{"{thread}"}/tasks/{"{id}"} — same EventBroker topic, defensive serialization</div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
