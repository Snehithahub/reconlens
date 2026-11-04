"""ReconLens dashboard for finance ops: run a reconciliation, review exceptions, export, estimate impact.

Run with ``streamlit run src/reconlens/ui/app.py --server.port 8503``. Talks to the API at ``API_URL``;
when that is unreachable (and RECONLENS_UI_MODE is not ``api``) it runs the API in-process.
"""

from __future__ import annotations

import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from reconlens.config import get_settings
from reconlens.money import fmt_inr
from reconlens.ui import data
from reconlens.ui.client import ApiError, ReconClient, Upload, connect

BAR_COLOR = "#2a78d6"
STATUS_LABELS = {"open": "Open", "in_progress": "In progress", "resolved": "Resolved", "ignored": "Ignored"}

st.set_page_config(page_title="ReconLens", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Connecting to the ReconLens API...")
def get_client(api_url: str, mode: str, local_db: str) -> ReconClient:
    return connect(api_url, mode, local_db=local_db)  # type: ignore[arg-type]


def client() -> ReconClient:
    mode = os.getenv("RECONLENS_UI_MODE", "auto").lower()
    local_db = os.getenv("RECONLENS_UI_LOCAL_DB", ":memory:")
    try:
        return get_client(get_settings().api_url, mode, local_db)
    except ApiError as e:
        st.error(f"Cannot reach the ReconLens API: {e.message}")
        if st.button("Retry"):
            st.rerun()
        st.stop()
        raise  # unreachable; st.stop() raises


def show_error(e: ApiError) -> None:
    st.error(e.message)
    if e.details:
        with st.expander("Details"):
            st.write(e.details)


# ---- state ------------------------------------------------------------------------------------
def load_run(c: ReconClient, run: dict[str, Any]) -> None:
    run_id = run["run_id"]
    st.session_state.run = run
    st.session_state.exc_items = c.exceptions(run_id)
    expl = c.explanations(run_id)
    if not expl and not st.session_state.get("llm_enabled"):
        expl = c.explain(run_id, use_llm=False)["items"]  # rule-based, instant
    st.session_state.explanations = expl


def replace_item(updated: dict[str, Any]) -> None:
    st.session_state.exc_items = [updated if i["exception_id"] == updated["exception_id"] else i
                              for i in st.session_state.exc_items]


# ---- sidebar ----------------------------------------------------------------------------------
def sidebar(c: ReconClient) -> None:
    sb = st.sidebar
    sb.title("ReconLens")
    sb.caption("Gateway settlements vs bank credits vs order ledger")
    try:
        ready = c.ready()
    except ApiError as e:
        ready = {"status": "unavailable", "llm_enabled": False, "gateway": "?"}
        sb.warning(e.message)
    st.session_state.llm_enabled = bool(ready.get("llm_enabled"))
    where = c.base_url if c.mode == "api" else "in-process (no API server found)"
    sb.caption(f"Backend: {where}  \nGateway: {ready.get('gateway')}  \n"
               f"LLM: {'configured' if st.session_state.llm_enabled else 'not configured (rule-based only)'}")

    sb.subheader("Data source")
    source = sb.radio("Data source", ["Use demo data", "Upload files"], label_visibility="collapsed")
    uploads: list[Any] = []
    if source == "Use demo data":
        n = sb.number_input("Payments to generate", min_value=100, max_value=200_000, value=5000, step=1000)
        seed = sb.number_input("Random seed", min_value=0, value=42, step=1)
        sb.caption("Synthetic PayFlow gateway report, bank statement and order ledger with injected discrepancies.")
    else:
        uploads = [
            sb.file_uploader("Gateway payments + settlement report", type=["csv", "xlsx"], key="up_gw"),
            sb.file_uploader("Bank statement", type=["csv", "xlsx"], key="up_bank"),
            sb.file_uploader("Order ledger", type=["csv", "xlsx"], key="up_ledger"),
        ]
    use_llm = sb.checkbox("Use LLM to read messy bank narrations", value=False,
                          disabled=not st.session_state.llm_enabled,
                          help="Only bank lines the rules cannot parse are sent to the LLM.")
    force = sb.checkbox("Re-run even if these inputs were reconciled before", value=False)

    if sb.button("Run reconciliation", type="primary", width="stretch"):
        try:
            with st.spinner("Reconciling..."):
                if source == "Use demo data":
                    run = c.run_demo(int(n), int(seed), use_llm=use_llm, force=force)
                elif all(uploads):
                    gw, bank, led = (Upload(u.name, u.getvalue()) for u in uploads)
                    run = c.run_upload(gw, bank, led, use_llm=use_llm, force=force)
                else:
                    sb.error("Upload all three files first.")
                    return
                load_run(c, run)
            st.toast("Reused the earlier run for these exact inputs." if run.get("reused") else "Run complete.")
        except ApiError as e:
            show_error(e)

    try:
        runs = c.list_runs(20)
    except ApiError:
        runs = []
    if runs:
        sb.subheader("Previous runs")
        labels = {r["run_id"]: f"{r['created_at'][:16].replace('T', ' ')}  {r['source']}" for r in runs}
        current = st.session_state.get("run", {}).get("run_id")
        pick = sb.selectbox("Open a run", list(labels), format_func=labels.get,
                            index=list(labels).index(current) if current in labels else None,
                            placeholder="Choose a run", label_visibility="collapsed")
        if pick and pick != current:
            try:
                load_run(c, c.get_run(pick))
                st.rerun()
            except ApiError as e:
                show_error(e)


# ---- views ------------------------------------------------------------------------------------
def bar_chart(df: pd.DataFrame, category: str, title: str, horizontal: bool, sort: list[str] | str
              ) -> alt.Chart:
    tooltip = [alt.Tooltip(f"{category}:N", title=title), alt.Tooltip("count:Q", title="Exceptions"),
               alt.Tooltip("impact_inr:Q", title="Impact (₹)", format=",.2f")]
    if horizontal:
        enc = {"y": alt.Y(f"{category}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=320)),
               "x": alt.X("count:Q", title="Exceptions", axis=alt.Axis(tickMinStep=1))}
        mark = {"height": {"band": 0.7}}
    else:
        enc = {"x": alt.X(f"{category}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0)),
               "y": alt.Y("count:Q", title="Exceptions", axis=alt.Axis(tickMinStep=1))}
        mark = {"width": {"band": 0.7}}
    height = max(160, 28 * len(df)) if horizontal else 260
    chart = alt.Chart(df).mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, **mark)
    return chart.encode(**enc, tooltip=tooltip).properties(height=height)


def overview(run: dict[str, Any], exc: pd.DataFrame) -> None:
    k = data.kpis(run["summary"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Auto-match rate", f"{k.auto_match_rate:.1%}", border=True,
              help=f"Payments fully reconciled across gateway, bank and ledger ({k.payments:,} payments).")
    c2.metric("Reconciled", data.inr_compact(k.reconciled_paise), border=True,
              help=fmt_inr(k.reconciled_paise) + " of captured payments")
    c3.metric("Unreconciled", data.inr_compact(k.unreconciled_paise), border=True,
              help=fmt_inr(k.unreconciled_paise) + " of captured payments with an exception or pending payout")
    c4.metric("Recoverable", data.inr_compact(k.recoverable_paise), border=True,
              help="Money the merchant can claim back (missing/short settlements, fee and GST overcharges, "
                   "duplicate refunds): " + fmt_inr(k.recoverable_paise))
    c5.metric("Fee + GST overcharges", data.inr_compact(k.fee_gst_recoverable_paise), border=True,
              help="MDR charged above the contracted rate plus GST mis-computed on fees: "
                   + fmt_inr(k.fee_gst_recoverable_paise))
    st.caption(f"Statement as of {k.as_of}. {k.settlements:,} settlements, "
               f"{k.settlement_auto_match_rate:.1%} matched to bank credits automatically. "
               f"{k.exceptions:,} exceptions.")

    unresolved_only = st.toggle("Only unresolved exceptions", value=True)
    view = exc[exc["status"].isin(["open", "in_progress"])] if unresolved_only and not exc.empty else exc
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Exceptions by type")
        bt = data.by_type(view)
        if bt.empty:
            st.info("No exceptions.")
        else:
            st.altair_chart(bar_chart(bt, "title", "Type", True, "-x"), width="stretch")
    with right:
        st.subheader("Exceptions by age")
        ba = data.by_ageing(view)
        st.altair_chart(bar_chart(ba, "ageing", "Age", False, list(ba["ageing"])), width="stretch")
        st.caption("Age is days from the exception's event date to the statement end date.")
    with st.expander("Table view"):
        st.dataframe(data.by_type(view).drop(columns="type"), hide_index=True, width="stretch",
                     column_config={"impact_inr": st.column_config.NumberColumn("Impact (₹)", format="%.2f")})


def evidence_view(item: dict[str, Any]) -> None:
    groups = data.evidence_by_source(item.get("evidence") or [])
    if not groups:
        st.caption("No evidence rows.")
    for label, frame in groups.items():
        st.markdown(f"**{label}**")
        st.dataframe(frame, hide_index=True, width="stretch")


def explanation_view(c: ReconClient, run_id: str, item: dict[str, Any]) -> None:
    expl = data.explanation_for(item["type"], st.session_state.get("explanations", []))
    if expl is None:
        if st.session_state.get("llm_enabled"):
            st.caption("AI explanations have not been generated for this run.")
            if st.button("Generate AI explanations", help="One LLM call per exception type."):
                try:
                    with st.spinner("Asking the LLM..."):
                        st.session_state.explanations = c.explain(run_id, use_llm=True)["items"]
                    st.rerun()
                except ApiError as e:
                    show_error(e)
        return
    st.markdown(f"**{data.explanation_label(expl)}**")
    st.write(expl.get("summary", ""))
    if expl.get("likely_cause"):
        st.markdown(f"Likely cause: {expl['likely_cause']}")
    if expl.get("draft_email"):
        with st.expander("Draft email to the gateway"):
            st.code(expl["draft_email"], language=None, wrap_lines=True)


def resolve_form(c: ReconClient, item: dict[str, Any]) -> None:
    with st.form(f"resolve_{item['exception_id']}"):
        cols = st.columns([1, 1])
        status = cols[0].selectbox("Status", data.STATUSES, format_func=STATUS_LABELS.get,
                                   index=data.STATUSES.index(item.get("status") or "open"))
        who = cols[1].text_input("Your name", value=item.get("resolved_by") or "", max_chars=120)
        note = st.text_area("Note", value=item.get("note") or "", max_chars=2000,
                            placeholder="e.g. Ticket PF-1234 raised with PayFlow on 4 Nov")
        if st.form_submit_button("Save"):
            try:
                replace_item(c.resolve(item["exception_id"], status, note.strip(), who.strip()))
                st.toast("Saved.")
                st.rerun()
            except ApiError as e:
                show_error(e)


def queue(c: ReconClient, run: dict[str, Any], exc: pd.DataFrame) -> None:
    if exc.empty:
        st.success("No exceptions in this run.")
        return
    titles = dict(zip(exc["type"], exc["title"], strict=False))
    f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
    types = f1.multiselect("Type", sorted(titles, key=titles.get), format_func=titles.get)
    sev = f2.multiselect("Severity", data.SEVERITY_ORDER, format_func=str.title)
    statuses = f3.multiselect("Status", data.STATUSES, default=["open", "in_progress"], format_func=STATUS_LABELS.get)
    ageing = f4.multiselect("Age", data.AGEING_ORDER)
    g1, g2, g3 = st.columns([3, 2, 2])
    search = g1.text_input("Search id, key or detail", placeholder="pay_..., setl_..., UTR")
    min_impact = g2.number_input("Min |impact| (₹)", min_value=0.0, value=0.0, step=100.0)
    recoverable = g3.checkbox("Recoverable only")
    view = data.filter_exceptions(exc, types=types, severities=sev, statuses=statuses, ageing=ageing,
                                  min_impact_inr=min_impact, recoverable_only=recoverable, search=search)
    st.session_state.queue_view = view
    st.caption(f"{len(view):,} of {len(exc):,} exceptions, total impact "
               f"{fmt_inr(int(view['impact_paise'].sum()))}")

    table = st.dataframe(
        view, hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row",
        key="queue_table",
        column_order=["severity", "title", "key", "impact_inr", "ageing", "status", "recoverable", "needs_review",
                      "event_date"],
        column_config={
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "title": st.column_config.TextColumn("Exception", width="medium"),
            "key": st.column_config.TextColumn("Reference"),
            "impact_inr": st.column_config.NumberColumn("Impact (₹)", format="%.2f"),
            "ageing": st.column_config.TextColumn("Age"),
            "status": st.column_config.TextColumn("Status"),
            "recoverable": st.column_config.CheckboxColumn("Recoverable"),
            "needs_review": st.column_config.CheckboxColumn("Needs review"),
            "event_date": st.column_config.DateColumn("Event date"),
        },
    )
    if view.empty:
        return
    rows = table.selection.rows if table is not None else []
    pos = rows[0] if rows and rows[0] < len(view) else 0
    item = view.iloc[pos].to_dict()
    if not rows:
        st.caption("Select a row to open it. Showing the first exception.")

    st.divider()
    st.subheader(item["title"])
    st.caption(f"{item['exception_id']}  ·  {item['severity'].title()} severity  ·  "
               f"{STATUS_LABELS.get(item['status'], item['status'])}"
               + (f"  ·  last updated by {item['resolved_by']}" if item.get("resolved_by") else ""))
    m1, m2, m3 = st.columns(3)
    m1.metric("Impact", fmt_inr(int(item["impact_paise"])))
    m2.metric("Age", "n/a" if pd.isna(item["age_days"]) else f"{int(item['age_days'])} days")
    m3.metric("Can be recovered", "Yes" if item["recoverable"] else "No")
    st.markdown(f"**What happened:** {item['detail']}")
    st.markdown(f"**Suggested action:** {item['suggested_action']}")
    if item.get("needs_review"):
        st.warning("Matched with lower confidence; confirm the pairing before accepting.")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Evidence")
        evidence_view(item)
    with right:
        st.markdown("##### Explanation")
        explanation_view(c, run["run_id"], item)
        st.markdown("##### Resolve or annotate")
        resolve_form(c, item)


def downloads(c: ReconClient, run: dict[str, Any], exc: pd.DataFrame) -> None:
    run_id = run["run_id"]
    st.subheader("Exceptions report")
    st.caption("All exceptions in this run with their current status and notes. The Excel file also has a "
               "settlements sheet.")
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Download CSV", lambda: c.report(run_id, "csv"), file_name=f"reconlens_{run_id}.csv",
                       mime="text/csv", width="stretch")
    d2.download_button("Download Excel", lambda: c.report(run_id, "xlsx"), file_name=f"reconlens_{run_id}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    view = st.session_state.get("queue_view")
    if view is not None and len(view) != len(exc):
        st.subheader("Filtered queue")
        st.caption(f"The {len(view):,} exceptions currently shown in the queue tab.")
        e1, e2, _ = st.columns([1, 1, 3])
        e1.download_button("Filtered CSV", data.queue_export(view, "csv"), file_name=f"reconlens_{run_id}_view.csv",
                           mime="text/csv", width="stretch")
        e2.download_button("Filtered Excel", data.queue_export(view, "xlsx"),
                           file_name=f"reconlens_{run_id}_view.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")


def impact(run: dict[str, Any]) -> None:
    st.subheader("Business impact (estimate)")
    st.caption("This is an estimate from your own assumptions, not a measurement. Adjust the assumptions to match "
               "how long your team takes today.")
    a1, a2, _ = st.columns([1, 1, 2])
    per_match = a1.number_input("Minutes per manual match", min_value=0.0, max_value=60.0, value=2.0, step=0.5,
                                help="Time to tick off one payment or one settlement against the bank by hand.")
    per_exc = a2.number_input("Minutes to investigate one exception", min_value=0.0, max_value=240.0, value=15.0,
                              step=5.0)
    est = data.impact_estimate(run["summary"], per_match, per_exc)
    m1, m2, m3 = st.columns(3)
    m1.metric("Items matched automatically", f"{est.auto_matched_units:,}",
              help="Fully reconciled payments plus settlements matched to bank credits.")
    m2.metric("Estimated hours saved per run", f"{est.hours_saved:,.1f}")
    m3.metric("Estimated hours of exception work left", f"{est.review_hours:,.1f}",
              help=f"{est.exceptions_to_review:,} exceptions x {per_exc:g} minutes")
    st.caption(f"Hours saved = {est.auto_matched_units:,} auto-matched items x {per_match:g} minutes / 60.")


def main() -> None:
    c = client()
    sidebar(c)
    run = st.session_state.get("run")
    if not run:
        st.title("Settlement reconciliation")
        st.write("Reconcile your payment gateway's settlement report against bank credits and your order ledger. "
                 "Choose demo data or upload the three files in the sidebar, then run the reconciliation.")
        st.markdown("- **Gateway report**: payments, refunds and chargebacks with fee, GST, settlement id and UTR\n"
                    "- **Bank statement**: date, narration and credit amount\n"
                    "- **Order ledger**: order id, amount and status")
        return
    st.title("Settlement reconciliation")
    st.caption(f"Run {run['run_id']}  ·  source {run['source']}  ·  created {run['created_at'][:19].replace('T', ' ')}"
               f"  ·  engine {run['summary'].get('engine_version', '?')}")
    exc = data.exceptions_frame(st.session_state.get("exc_items", []))
    t1, t2, t3, t4 = st.tabs(["Overview", "Exception queue", "Download", "Business impact"])
    with t1:
        overview(run, exc)
    with t2:
        queue(c, run, exc)
    with t3:
        downloads(c, run, exc)
    with t4:
        impact(run)


main()
