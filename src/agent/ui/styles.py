"""Visual system for the Streamlit engineering console."""

APP_CSS = """
<style>
:root {
  --ink: #17211b;
  --muted: #66736b;
  --line: #d8dfda;
  --surface: #ffffff;
  --canvas: #f3f5f2;
  --green: #16734a;
  --amber: #a85c00;
  --red: #a73535;
}

.stApp { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(243,245,242,.92); }
[data-testid="stSidebar"] { background: #e9eeea; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
.block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 2rem; }

h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
h1 { font-size: 1.65rem !important; line-height: 1.25 !important; }
h2 { font-size: 1.05rem !important; }
h3 { font-size: .94rem !important; }
p, label, [data-testid="stMarkdownContainer"] { letter-spacing: 0; }

[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: .7rem .8rem;
  min-height: 84px;
}
[data-testid="stMetricValue"] { font-size: 1.25rem; }
[data-testid="stChatMessage"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: .8rem 1rem;
}
[data-testid="stChatInput"] { border-color: var(--line); }
.stButton > button { border-radius: 5px; min-height: 38px; }
.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
.stTextInput input, .stSelectbox [data-baseweb="select"] { border-radius: 5px; }

.agent-status {
  display: grid;
  grid-template-columns: 10px 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 5px 0;
  font-size: .86rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #97a39b; }
.status-dot.running { background: var(--green); }
.status-dot.waiting { background: var(--amber); }
.status-label { color: var(--muted); font-size: .76rem; }
.event-line {
  border-left: 2px solid var(--line);
  margin-left: 6px;
  padding: 2px 0 12px 14px;
}
.event-time { color: var(--muted); font-size: .74rem; }
.event-name { color: var(--ink); font-weight: 600; font-size: .86rem; }
.approval-band {
  border: 1px solid #d6a45f;
  border-left: 4px solid var(--amber);
  background: #fff8ec;
  border-radius: 5px;
  padding: .75rem 1rem;
  margin: .5rem 0;
}
.repo-meta { color: var(--muted); font-size: .78rem; overflow-wrap: anywhere; }

@media (max-width: 760px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: 1rem; }
  h1 { font-size: 1.35rem !important; }
  [data-testid="stHorizontalBlock"] { gap: .45rem; }
  [data-testid="stMetric"] { min-height: 74px; padding: .55rem; }
  [data-testid="stMetricValue"] { font-size: 1rem; }
}
</style>
"""
