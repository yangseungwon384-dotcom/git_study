"""SafeZoo 왼쪽 메뉴 화면.

홈, 카메라, 알림, 이벤트 로그, 스냅샷, AI 보조 판단, 설정을 나눈다.
확인·보관 결과는 DB에 남고, 미리보기 대상 ID는 이 브라우저 세션의 상태에만 둔다.
"""
import html
import json
import time

from dashboard_policy import END_REASON_KO, DemoGate
from dashboard_store import evidence_status


DASHBOARD_CSS = """
:root, html[data-theme="dark"] {
  --sz-bg: #141414;
  --sz-sidebar: #101010;
  --sz-card: #232323;
  --sz-text: #ffffff;
  --sz-muted: #d4d4d4;
  --sz-line: #3a3a3a;
  --sz-focus: #ffffff;
  --sz-btn: #2b2b2b;
  --sz-btn-text: #ffffff;
  --sz-input: #1a1a1a;
  --sz-accent: #e50914;
  --sz-accent-text: #ffffff;
}
html[data-theme="light"] {
  --sz-bg: #f3f3f3;
  --sz-sidebar: #ebebeb;
  --sz-card: #ffffff;
  --sz-text: #141414;
  --sz-muted: #333333;
  --sz-line: #cfcfcf;
  --sz-focus: #141414;
  --sz-btn: #ffffff;
  --sz-btn-text: #141414;
  --sz-input: #ffffff;
  --sz-accent: #e50914;
  --sz-accent-text: #ffffff;
}
html, body, .gradio-container, .app, .contain, footer {
  background: var(--sz-bg) !important;
  color: var(--sz-text) !important;
  font-family: "Pretendard", "Malgun Gothic", "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  padding: 12px 16px 24px !important;
}
.gradio-container p, .gradio-container span, .gradio-container label, .gradio-container li,
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container strong,
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose li,
.gradio-container .markdown, .gradio-container td, .gradio-container th {
  color: var(--sz-text) !important;
}
.gradio-container .block, .gradio-container .form, .gradio-container .panel,
.gradio-container .wrap, #main .block {
  background: var(--sz-card) !important;
  border: 1px solid var(--sz-line) !important;
  border-radius: 4px !important;
  color: var(--sz-text) !important;
}
.gradio-container label > span, .gradio-container .label-wrap span {
  color: var(--sz-muted) !important;
  background: transparent !important;
}
/* Gradio 선택 칸은 밝은 배경이다. 그 안의 글자는 검게 둔다. */
.gradio-container label:has(> input),
.gradio-container label:has(> input) span,
.gradio-container label:has(> input) p,
.gradio-container .options,
.gradio-container .options li,
.gradio-container .options li span,
.gradio-container [role="listbox"],
.gradio-container [role="listbox"] span,
.gradio-container [role="option"],
.gradio-container [role="option"] span,
.gradio-container [role="menuitem"],
.gradio-container [role="menuitem"] span {
  color: #141414 !important;
  -webkit-text-fill-color: #141414 !important;
}
.gradio-container label:has(> input),
.gradio-container .options,
.gradio-container .options li,
.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container [role="menuitem"] {
  background: #f3f3f3 !important;
  border-color: #d0d0d0 !important;
}
.gradio-container button,
.gradio-container button.primary,
.gradio-container button.secondary,
.gradio-container button[variant="primary"] {
  background: #ffffff !important;
  color: #141414 !important;
  -webkit-text-fill-color: #141414 !important;
  border: 1px solid #cfcfcf !important;
  border-radius: 4px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
.gradio-container button span, .gradio-container button p,
.gradio-container button.primary span, .gradio-container button.primary p,
.gradio-container button.secondary span, .gradio-container button.secondary p {
  background: transparent !important;
  color: #141414 !important;
  -webkit-text-fill-color: #141414 !important;
}
.gradio-container input:not([type="radio"]):not([type="checkbox"]),
.gradio-container textarea,
.gradio-container select,
.gradio-container .secondary-wrap,
.gradio-container .secondary-wrap input {
  background: #ffffff !important;
  color: #141414 !important;
  -webkit-text-fill-color: #141414 !important;
  border: 1px solid #c8c8c8 !important;
  border-radius: 10px !important;
}
.gradio-container input::placeholder, .gradio-container textarea::placeholder {
  color: #595959 !important;
  -webkit-text-fill-color: #595959 !important;
  opacity: 1 !important;
}
.gradio-container table, .gradio-container [role="grid"],
.gradio-container th, .gradio-container td,
.gradio-container [role="columnheader"], .gradio-container [role="gridcell"] {
  background: var(--sz-card) !important;
  color: var(--sz-text) !important;
  border-color: var(--sz-line) !important;
}
.gradio-container th, .gradio-container [role="columnheader"] {
  background: var(--sz-sidebar) !important;
  color: var(--sz-text) !important;
}
#shell { align-items: stretch; gap: 0; min-height: 100vh; }
#sidebar {
  min-width: 220px;
  width: 220px;
  background: var(--sz-sidebar) !important;
  border: none !important;
  border-right: 1px solid var(--sz-line) !important;
  border-radius: 0 !important;
  padding: 22px 16px !important;
}
#sidebar.collapsed { display: none; }
#sidebar button {
  width: 100%;
  text-align: left;
  margin: 2px 0;
  padding: 9px 10px !important;
  background: transparent !important;
  border: none !important;
  color: var(--sz-text) !important;
  -webkit-text-fill-color: var(--sz-text) !important;
}
#sidebar button span, #sidebar button p {
  color: var(--sz-text) !important;
  -webkit-text-fill-color: var(--sz-text) !important;
}
#main { padding: 8px 18px 24px !important; background: transparent !important; border: none !important; }
#topbar { align-items: center; border-bottom: 1px solid var(--sz-line); margin-bottom: 14px; }
.sz-logo {
  color: var(--sz-accent) !important;
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -1.4px;
  margin: 0 0 18px;
}
.sz-title { font-size: 28px; font-weight: 750; letter-spacing: -0.8px; margin: 8px 0; color: var(--sz-text) !important; }
.sz-pill, .sz-clock, .sz-ai {
  color: var(--sz-text) !important;
  font-size: 14px;
}
.sz-kicker { color: var(--sz-muted) !important; font-size: 13px; margin: 0 0 8px; letter-spacing: 0.04em; }
.sz-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 0 0 14px; }
.sz-stats div { background: var(--sz-card); border: 1px solid var(--sz-line); padding: 12px 14px; }
.sz-stats span { display: block; color: var(--sz-muted) !important; font-size: 13px; }
.sz-stats strong { display: block; color: var(--sz-text) !important; font-size: 26px; margin-top: 4px; }
.sz-warn { color: #ff8d8d !important; }
#home-split { align-items: stretch; }
#alert-side { border-left: 1px solid var(--sz-line) !important; padding-left: 12px; }
#sidebar { background: var(--sz-sidebar) !important; }
#main, #page-home, #home-split, #stage, #alert-side, #topbar {
  background: transparent !important;
  box-shadow: none !important;
}
#main, #page-home, #home-split, #stage, #topbar { border: none !important; }
#ai-foot { margin-top: auto !important; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--sz-focus) !important;
  outline-offset: 2px;
}
html[data-density="compact"] .block { padding: 4px !important; }
.sz-card { font-size: 16px; line-height: 1.5; }
table, [role="grid"] { display: block; overflow-x: auto; max-width: 100%; }
@media (max-width: 900px) {
  #shell { display: block !important; }
  #sidebar { width: 100%; min-width: 0; border-right: none !important; }
  .sz-stats { grid-template-columns: 1fr 1fr; }
  #home-split, #alert-side { display: block !important; border-left: none !important; padding-left: 0; }
}
"""

THEME_JS = """
(theme) => {
  const mode = theme === '밝게' ? 'light' : 'dark';
  document.documentElement.dataset.theme = mode;
  try { localStorage.setItem('safezoo-theme', mode); } catch (e) {}
}
"""
THEME_LOAD_JS = """
() => {
  let mode = 'dark';
  try { mode = localStorage.getItem('safezoo-theme') || 'dark'; } catch (e) {}
  document.documentElement.dataset.theme = mode;
  return mode === 'light' ? '밝게' : '다크';
}
"""
DENSITY_JS = """
(density) => {
  const mode = density === '좁게' ? 'compact' : 'comfortable';
  document.documentElement.dataset.density = mode;
  try { localStorage.setItem('safezoo-density', mode); } catch (e) {}
}
"""


def build_dashboard(original, engine):
    import cv2
    import gradio as gr
    import numpy as np
    import pandas as pd
    from zoo_vision_jev_new import (
        CLASS_KO, STATUS_KO, jev_brief, paint_clicks, polygon_crosses, questions, reference_label)

    original.ANALYZER = engine
    original.refresh_detect = engine.step_all
    original.load_events = lambda: pd.DataFrame(engine.db.events(), columns=original.EVENT_COLS)
    engine.focus_index = 0

    def provider_line():
        if not engine.args.jev:
            return 'JSON 준비 모드 · 외부 모델 호출 없음'
        if getattr(engine.args, 'provider', '') == 'ollama':
            return f'Ollama TEV · 모델 {engine.args.jev_model}'
        return f'TypeSafe JEV · 모델 {engine.args.jev_model}'

    def progress(item):
        if item.get('active'):
            return '진행 중'
        return END_REASON_KO.get(item.get('end_reason') or '', item.get('end_reason') or '종료')

    def ack_label(item):
        return '확인함' if item.get('acknowledged_at') else '미확인'

    def table_rows(records):
        rows = []
        for item in records:
            rows.append([
                item['id'], item.get('detected_at', ''), item.get('camera_id', ''),
                item.get('object', ''), item.get('event_type', ''), item.get('severity', ''),
                progress(item), ack_label(item),
                '' if item.get('active') else END_REASON_KO.get(item.get('end_reason') or '', item.get('end_reason') or ''),
            ])
        return rows

    def ack_hold():
        holds = getattr(engine, 'ack_hold', None)
        if holds is None:
            holds = {}
            engine.ack_hold = holds
        return holds

    def alert_checked(item):
        holds = ack_hold()
        held = holds.get(item['id'])
        if held and time.monotonic() - held[1] < 4:
            return held[0]
        if held:
            holds.pop(item['id'], None)
        return bool(item.get('acknowledged_at'))

    def alert_rows(records):
        rows = []
        for item in records:
            rows.append([
                alert_checked(item), item['id'], item.get('detected_at', ''), item.get('camera_id', ''),
                item.get('object', ''), item.get('event_type', ''), item.get('severity', ''),
            ])
        return rows

    def as_checked(value):
        return value is True or str(value).strip().lower() in ('true', '1', 'yes')

    def frame_rows(value):
        if value is None:
            return []
        if hasattr(value, 'values'):
            return value.values.tolist()
        return list(value)

    def apply_alert_checks(frame):
        notes = []
        holds = ack_hold()
        for row in frame_rows(frame):
            if not row or len(row) < 2:
                continue
            event_id = str(row[1] or '').strip()
            if not event_id:
                continue
            checked = as_checked(row[0])
            record = engine.db.record(event_id)
            if not record:
                continue
            if checked == alert_checked(record):
                continue
            held = holds.get(event_id)
            if held and time.monotonic() - held[1] < 4 and checked != held[0]:
                continue
            holds[event_id] = (checked, time.monotonic())
            if checked:
                engine.db.acknowledge([event_id], '')
                notes.append('확인')
            else:
                engine.db.unacknowledge([event_id])
                notes.append('확인 취소')
        if not notes:
            return ''
        return f"확인 {notes.count('확인')}건, 확인 취소 {notes.count('확인 취소')}건"

    def summary():
        linked = analysis_ok = 0
        with engine.lock:
            cameras = list(engine.runtime.values())
        for runtime in cameras:
            age = time.monotonic() - runtime['stamp']
            video_ok = runtime['frame'] is not None and age <= engine.args.max_gap
            if video_ok:
                linked += 1
            broken = bool(runtime['error']) or not video_ok
            if not broken and engine.analyze_thread and engine.analyze_thread.is_alive():
                analysis_ok += 1
        active = engine.db.active_count()
        unacked = engine.db.unacked_count()
        broken = any(runtime['error'] or time.monotonic() - runtime['stamp'] > engine.args.max_gap
                     for runtime in cameras)
        note = ''
        if broken:
            note = '<p class="sz-warn">오류 또는 영상 끊김이 있습니다. 전체를 안전으로 보지 않습니다.</p>'
        elif unacked == 0:
            note = '<p class="sz-kicker">미확인 알림 없음</p>'
        head = (
            '<div class="sz-stats">'
            f'<div><span>카메라 상태</span><strong>{linked}/{len(cameras) or 4}</strong></div>'
            f'<div><span>진행 경고</span><strong>{active}</strong></div>'
            f'<div><span>미확인 알림</span><strong>{unacked}</strong></div>'
            f'<div><span>분석 상태</span><strong>{analysis_ok}/{len(cameras) or 4}</strong></div>'
            f'</div>{note}'
        )
        return (head, f'<span class="sz-pill">미확인 {unacked}</span>',
                f'<div class="sz-ai">AI: {html.escape(provider_line())}</div>',
                f'<span class="sz-clock">{time.strftime("%H:%M:%S")}</span>')

    def home_alerts():
        rows = engine.db.list_records(ack='open', archived='visible', limit=8)
        return alert_rows(rows) or [[False, '', '', '', '', '', '미확인 알림 없음']]

    def filtered(day_from, day_to, camera, severity, kind, ack_name, archived):
        ack = {'미확인': 'open', '확인함': 'acked'}.get(ack_name, 'all')
        return engine.db.list_records(day_from or None, day_to or None, camera, severity, kind, ack, archived)

    def refresh_alerts(camera, severity, ack_name):
        return alert_rows(filtered(None, None, camera, severity, '전체', ack_name, 'visible'))

    def refresh_logs(day_from, day_to, camera, severity, kind, ack_name, view):
        archived = 'archived' if view == '보관된 로그' else 'visible'
        rows = filtered(day_from, day_to, camera, severity, kind, ack_name, archived)
        return table_rows(rows), f'현재 조회 {len(rows)}건 · 발생 기준(보관 포함) {engine.db.occurrence_count()}건'

    def ai_for(event_id):
        if not event_id:
            return '사건을 선택하면 그 사건의 보조 판단만 표시합니다.'
        rows = engine.db.decisions_for(event_id)
        if not rows:
            return '이 사건의 보조 판단 기록이 없습니다. 다른 사건의 답을 여기 넣지 않습니다.'
        latest = rows[0]
        valid = next((item for item in rows if item.get('status') == 'completed' and item.get('result_json')), None)
        chunks = [f"요청 중 상태: {STATUS_KO.get(latest.get('status'), latest.get('status'))} · {latest.get('requested_at')}"]
        if latest.get('error'):
            chunks.append('오류: ' + latest['error'])
        if valid:
            response = json.loads(valid['result_json'])
            state = json.loads(valid['state_json']) if valid.get('state_json') else {}
            headline, lines, *_rest = jev_brief(state, valid.get('status'), response.get('answers') or {})
            ref = reference_label(valid, engine.args.jev_ttl)
            chunks.append(f"마지막 유효 결과 · {ref} · {valid.get('requested_at')}")
            chunks.append(headline)
            chunks.extend(lines)
        else:
            chunks.append('마지막 유효 결과 없음')
        return '\n'.join(chunks)

    def detail(event_id):
        item = engine.db.record(event_id) if event_id else None
        if not item:
            return '사건을 선택하세요. 상세를 열어도 확인 처리되지 않습니다.', None, ''
        note, path = evidence_status(item.get('snapshot_path'))
        if item.get('snapshot_archived_at'):
            note = (note + '\n' if note else '') + '스냅샷은 갤러리에서 보관됨. 파일과 사건 기록은 남아 있습니다.'
        text = '\n'.join([
            f"사건 {item['id']}",
            f"카메라 {item.get('camera_id')} · {item.get('detected_at')}",
            f"객체 {item.get('object')} · {item.get('event_type')} · {item.get('severity')}",
            f"확인 {ack_label(item)} · 확인시각 {item.get('acknowledged_at') or '-'}",
            f"표시명 {item.get('ack_name') or '-'}",
            f"검토 {item.get('review_label') or '없음'} {item.get('review_note') or ''}",
            note,
            ai_for(event_id),
        ])
        return text, path, item.get('review_note') or ''

    def selected_id(evt: gr.SelectData, id_at=0):
        row = getattr(evt, 'row_value', None)
        if isinstance(row, (list, tuple)) and len(row) > id_at:
            return row[id_at] or ''
        if isinstance(evt.value, (list, tuple)) and len(evt.value) > id_at:
            return evt.value[id_at]
        return ''

    def ack_one(event_id, display_name):
        if not event_id:
            return '사건 ID가 없습니다.'
        result = engine.db.acknowledge([event_id], display_name)
        return f"확인 {len(result['changed'])}건, 이미 확인 {len(result['unchanged'])}건, 없음 {len(result['missing'])}건"

    def ack_many(ids, display_name):
        result = engine.db.acknowledge(ids or [], display_name)
        return (f"확인 {len(result['changed'])}건, 이미 확인 {len(result['unchanged'])}건, "
                f"없음 {len(result['missing'])}건. ID {', '.join(result['changed']) or '-'}")

    def add_id(ids, event_id):
        ids = list(ids or [])
        if event_id and event_id not in ids:
            ids.append(event_id)
        return ids, '선택 ID: ' + (', '.join(ids) if ids else '없음')

    def unack(event_id):
        if not event_id:
            return '사건 ID가 없습니다.'
        result = engine.db.unacknowledge([event_id])
        return f"확인 취소 {len(result['changed'])}건"

    def save_review(event_id, label, note):
        if not event_id:
            return '사건을 선택한 뒤 검토를 저장하세요.'
        engine.db.review(event_id, label, note)
        return '검토를 저장했습니다. 자동 학습이나 제외 구역 생성은 하지 않습니다.'

    def preview_archive(day_from, day_to, camera, severity, kind, ack_name, target):
        rows = filtered(day_from, day_to, camera, severity, kind, ack_name,
                        'visible' if target == 'log' else 'all')
        column = 'log_archived_at' if target == 'log' else 'snapshot_archived_at'
        picked = [item for item in rows if not item['active'] and item.get('acknowledged_at') and not item.get(column)]
        ids = [item['id'] for item in picked]
        cameras = sorted({item.get('camera_id', '') for item in picked})
        days = sorted(item.get('detected_at', '')[:10] for item in picked if item.get('detected_at'))
        period = f'{days[0]} ~ {days[-1]}' if days else '-'
        text = f"대상 {len(ids)}건 · 카메라 {', '.join(cameras) or '-'} · 기간 {period}"
        if not ids:
            text = '보관할 대상이 없습니다. 종료됐고 확인된 사건만 대상입니다.'
        return ids, text

    def run_archive(ids, target):
        if not ids:
            return '보관할 대상이 없습니다.', []
        result = engine.db.archive(ids, target)
        text = f"보관 {len(result['archived'])}건, 제외 {len(result['skipped'])}건"
        if result['skipped']:
            text += ' (자격 변경·이미 보관·진행 중·미확인)'
        return text, ids

    def run_restore(event_id, target):
        if not event_id:
            return '복원할 사건 ID가 없습니다.'
        result = engine.db.restore([event_id], target)
        return f"복원 {len(result['restored'])}건, 제외 {len(result['skipped'])}건"

    def shot_paths(day_from, day_to, camera, kind):
        rows = engine.db.list_records(day_from or None, day_to or None, camera, None, kind, 'all', 'gallery', limit=60)
        items = []
        for item in rows:
            note, path = evidence_status(item.get('snapshot_path'))
            items.append(dict(id=item['id'], path=path, caption=f"{item.get('camera_id')} {item.get('detected_at')} {ack_label(item)}",
                              missing=note))
        return items

    def shot_view(day_from, day_to, camera, kind, index):
        items = shot_paths(day_from, day_to, camera, kind)
        gallery = [(item['path'], item['caption']) for item in items if item['path']]
        missing = [item['caption'] + ' · 증거 이미지 없음 · 사건 ' + item['id'] for item in items if not item['path']]
        if not items:
            return gallery, None, '사진이 없습니다.', 0, ''
        index = max(0, min(int(index or 0), len(items) - 1))
        current = items[index]
        note, path = evidence_status(current['path'])
        text = f"사건 {current['id']} · {current['caption']}"
        if note:
            text += '\n증거 이미지 없음'
        return gallery, path, text + ('\n' + '\n'.join(missing) if missing else ''), index, current['id']

    def move_shot(day_from, day_to, camera, kind, index, step):
        items = shot_paths(day_from, day_to, camera, kind)
        if not items:
            return 0
        return (int(index or 0) + step) % len(items)

    def video_tick():
        frames = engine.step_all()
        while len(frames) < 4:
            frames.append(None)
        frames = frames[:4]
        focus = frames[getattr(engine, 'focus_index', 0) % 4]
        return (*frames, *frames, focus)

    def set_focus(code):
        codes = [cam['code'] for cam in original.CAMERAS]
        engine.focus_index = codes.index(code) if code in codes else 0
        return code

    def camera_label(code):
        cam = next((item for item in original.CAMERAS if item['code'] == code), original.CAMERAS[0])
        return f"{cam['code']} · {cam['title']}"

    def show_one(code):
        """선택한 카메라만 크게 보여 준다. 4분할로 바꾸면 네 화면이 다시 나온다."""
        set_focus(code)
        hidden = tuple(gr.update(visible=False) for _ in original.CAMERAS)
        shown = gr.update(visible=True, label=camera_label(code))
        return ('1분할', *hidden, shown)

    def show_roi(value):
        with engine.lock:
            for runtime in engine.runtime.values():
                runtime['show_roi'] = bool(value)
        return 'ROI 표시를 바꿨습니다. 감시 활성 설정은 그대로입니다.'

    def roi_frame(label):
        cam = next(item for item in original.CAMERAS if item['code'] == label)
        runtime = engine.runtime[cam['cam_id']]
        with engine.lock:
            frame = None if runtime['frame'] is None else runtime['frame'].copy()
            cfg = dict(runtime['cfg'] or {})
        return frame, cfg

    def draw(label, points):
        frame, _cfg = roi_frame(label)
        if frame is None:
            return None
        return paint_clicks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), points)

    def roi_open(label):
        frame, cfg = roi_frame(label)
        if frame is None:
            return None, [], [], False, False, False, False, '영상이 준비되지 않았습니다.'
        height, width = frame.shape[:2]
        safety = [[float(x) * (width - 1), float(y) * (height - 1)] for x, y in cfg.get('polygon') or []]
        exclude = [[float(x) * (width - 1), float(y) * (height - 1)] for x, y in cfg.get('exclude_polygon') or []]
        return (draw(label, safety), safety, exclude, bool(cfg.get('enabled')) if safety else True,
                bool(cfg.get('person_inside')), bool(cfg.get('reviewed')), bool(cfg.get('exclude_reviewed')),
                f'안전구역 점 {len(safety)}개, 제외 구역 점 {len(exclude)}개. 저장 전까지 감시 구역은 바뀌지 않습니다.')

    def roi_click(evt, safety, exclude, label, mode):
        points = list(exclude if mode == '탐지 제외 구역' else safety or [])
        frame, _cfg = roi_frame(label)
        if frame is None:
            return None, safety or [], exclude or [], '영상이 준비되지 않았습니다.'
        height, width = frame.shape[:2]
        idx = getattr(evt, 'index', None)
        if not isinstance(idx, (list, tuple)) or len(idx) < 2:
            return draw(label, points), safety or [], exclude or [], '클릭 위치를 읽지 못했습니다.'
        x, y = float(idx[0]), float(idx[1])
        if not (0 <= x < width and 0 <= y < height):
            return draw(label, points), safety or [], exclude or [], '화면 안쪽을 클릭하세요.'
        points.append([x, y])
        if mode == '탐지 제외 구역':
            exclude = points
        else:
            safety = points
        note = f'{mode} 점 {len(points)}개.'
        if len(points) >= 3 and polygon_crosses(points):
            note += ' 선이 교차합니다.'
        return draw(label, points), safety, exclude, note

    def roi_undo(safety, exclude, label, mode):
        if mode == '탐지 제외 구역':
            exclude = list(exclude or [])[:-1]
            points = exclude
        else:
            safety = list(safety or [])[:-1]
            points = safety
        return draw(label, points), safety, exclude, '마지막 점을 취소했습니다. 저장된 구역은 그대로입니다.'

    def roi_reset(label):
        return roi_open(label)

    def roi_save(label, safety, exclude, enabled, person_inside, reviewed, exclude_reviewed):
        cam = next(item for item in original.CAMERAS if item['code'] == label)
        runtime = engine.runtime[cam['cam_id']]
        try:
            with engine.lock:
                if runtime['frame'] is None:
                    raise ValueError('영상이 준비되지 않았습니다.')
                height, width = runtime['frame'].shape[:2]
                polygon = _checked_polygon(safety, width, height, required=True)
                excluded = _checked_polygon(exclude, width, height, required=False)
                version = runtime['roi_version'] + 1
                cfg = dict(polygon=polygon, enabled=bool(enabled), person_inside=bool(person_inside),
                           reviewed=bool(reviewed), version=version,
                           exclude_polygon=excluded, exclude_reviewed=bool(exclude_reviewed and excluded))
                with engine.db.connect() as db:
                    db.execute(
                        'INSERT INTO roi(camera, json) VALUES(?,?) '
                        'ON CONFLICT(camera) DO UPDATE SET json=excluded.json',
                        (cam['cam_id'], json.dumps(cfg)))
                engine.reset(runtime, 'roi_changed')
                runtime['cfg'] = cfg
                runtime['roi_version'] = version
            return 'ROI를 저장했습니다. 이 카메라의 이전 추적과 진행 중 판단은 새 구역에 재사용하지 않습니다.'
        except (ValueError, TypeError) as exc:
            return str(exc)

    def _checked_polygon(points, width, height, required):
        points = list(points or [])
        if len(points) < 3:
            if required:
                raise ValueError('안전구역은 가장자리를 3곳 이상 클릭한 뒤 저장하세요.')
            return []
        if polygon_crosses(points):
            raise ValueError('선이 교차합니다. 마지막 점을 취소하고 가장자리 순서로 다시 찍으세요.')
        polygon = np.array([[int(round(point[0])), int(round(point[1]))] for point in points], np.int32)
        if abs(cv2.contourArea(polygon.astype(np.float32))) < 10:
            raise ValueError('구역이 너무 좁습니다.')
        return (polygon / [width - 1, height - 1]).tolist()

    def save_gate(code, class_name, candidate, confirm, samples, span, gap, boundary, safe_return):
        cam = next(item for item in original.CAMERAS if item['code'] == code)
        if not class_name:
            return '클래스를 입력하세요. 예: giraffe, person'
        gate = DemoGate(candidate_min=float(candidate), confirm_min=float(confirm),
                        min_samples=int(samples), min_span_sec=float(span), max_gap_sec=float(gap),
                        boundary_norm=float(boundary), safe_return_sec=float(safe_return))
        version = engine.db.save_gate(cam['cam_id'], class_name.strip(), gate)
        engine.gates[(cam['cam_id'], class_name.strip())] = gate
        return (f'{cam["title"]}의 {class_name}에만 저장했습니다. 버전 {version}. '
                '시연용 기준이며 현장 검증값이 아닙니다. 단위: 신뢰도 0~1, 시간 초, '
                '완충 폭은 프레임 짧은 변 대비 비율.')

    def ai_page(event_id):
        records = engine.db.latest()
        latest = records[0] if records else None
        request_state = '최근 요청 없음' if latest is None else (
            f"최근 요청 {STATUS_KO.get(latest.get('status'), latest.get('status'))} · {latest.get('requested_at')}")
        body = '<div class="sz-card"><p>' + html.escape(provider_line()) + '</p><p>' + html.escape(request_state) + '</p>'
        if latest and latest.get('result_json'):
            response = json.loads(latest['result_json'])
            state = json.loads(latest['state_json']) if latest.get('state_json') else {}
            headline, lines, *_rest = jev_brief(state, latest.get('status'), response.get('answers') or {})
            body += '<h2>' + html.escape(headline) + '</h2><ul>' + ''.join(
                f'<li>{html.escape(line)}</li>' for line in lines) + '</ul>'
        body += '<p>위는 가장 최근 요청입니다. 아래는 선택한 사건만의 기록입니다.</p><pre>' + html.escape(ai_for(event_id)) + '</pre></div>'
        payload = {}
        received = {}
        if latest:
            payload = {'model': engine.args.jev_model, 'state': json.loads(latest['state_json']),
                       'questions': questions(engine.args.score)}
            if latest.get('result_json'):
                received = json.loads(latest['result_json'])
        return body, payload, received

    headers = ['사건 ID', '시각', '카메라', '객체', '종류', '심각도', '진행', '확인', '종료 사유']
    alert_headers = ['확인', '사건 ID', '시각', '카메라', '객체', '종류', '심각도']
    alert_types = ['bool', 'str', 'str', 'str', 'str', 'str', 'str']
    cameras = ['전체'] + [cam['cam_id'] for cam in original.CAMERAS]
    codes = [cam['code'] for cam in original.CAMERAS]
    with gr.Blocks(title='SafeZoo') as demo:
        with gr.Row(elem_id='shell'):
            with gr.Column(elem_id='sidebar', scale=0, min_width=220):
                gr.HTML('<div class="sz-logo">SafeZoo</div>')
                home_btn = gr.Button('홈')
                cam_btn = gr.Button('카메라')
                alert_btn = gr.Button('알림')
                log_btn = gr.Button('이벤트 로그')
                shot_btn = gr.Button('스냅샷')
                ai_btn = gr.Button('AI 보조 판단')
                set_btn = gr.Button('설정')
                fold = gr.Button('메뉴 접기')
                provider = gr.HTML('<div class="sz-ai">AI: 연결 상태</div>', elem_id='ai-foot')
            with gr.Column(elem_id='main'):
                with gr.Row(elem_id='topbar'):
                    title_html = gr.HTML('<div class="sz-title">홈</div>')
                    badge = gr.HTML('<span class="sz-pill">미확인 0</span>')
                    theme = gr.Radio(['다크', '밝게'], value='다크', label='테마', scale=0)
                    clock = gr.HTML('<span class="sz-clock">--:--:--</span>')
                with gr.Column(visible=True, elem_id='page-home') as page_home:
                    counts = gr.HTML()
                    with gr.Row(elem_id='home-split'):
                        with gr.Column(scale=3, elem_id='stage'):
                            gr.HTML('<p class="sz-kicker">주요 카메라 영상</p>')
                            with gr.Row():
                                home_imgs = [gr.Image(label=cam['title'], height=210, interactive=False,
                                                      format='jpeg') for cam in original.CAMERAS[:2]]
                            with gr.Row():
                                home_imgs += [gr.Image(label=cam['title'], height=210, interactive=False,
                                                       format='jpeg') for cam in original.CAMERAS[2:]]
                            with gr.Row():
                                jump_buttons = [gr.Button(cam['title'] + ' 상세') for cam in original.CAMERAS]
                        with gr.Column(scale=1, elem_id='alert-side'):
                            gr.HTML('<p class="sz-kicker">미확인 알림</p>')
                            home_table = gr.Dataframe(
                                headers=alert_headers, datatype=alert_types, interactive=True,
                                static_columns=[1, 2, 3, 4, 5, 6], column_count=7, label='카메라 · 시각 · 경고')
                            home_msg = gr.Markdown()
                            home_detail = gr.Button('상세')
                    gr.HTML('<p class="sz-kicker">최근 사건 · 스냅샷 미리보기</p>')
                    home_gallery = gr.Gallery(label='최근 스냅샷', columns=4, height=150, allow_preview=False)
                with gr.Column(visible=False) as page_cam:
                    with gr.Row():
                        cam_pick = gr.Dropdown(codes, value=codes[0], label='카메라')
                        layout = gr.Radio(['4분할', '1분할'], value='4분할', label='분할')
                        roi_visible = gr.Checkbox(value=True, label='ROI 표시')
                    cam_imgs = [gr.Image(label=cam['title'], height=240, interactive=False, format='jpeg')
                                for cam in original.CAMERAS]
                    large = gr.Image(label='선택 카메라', height=520, interactive=False, format='jpeg',
                                     visible=False, buttons=['fullscreen'])
                    cam_note = gr.Markdown()
                    safety_points = gr.State([])
                    exclude_points = gr.State([])
                    with gr.Accordion('구역 설정', open=False):
                        mode = gr.Radio(['안전구역', '탐지 제외 구역'], value='안전구역', label='편집 대상')
                        load_frame = gr.Button('현재 화면 불러오기')
                        board = gr.Image(label='가장자리를 클릭하세요', type='numpy', interactive=False,
                                         height=420, buttons=['fullscreen'])
                        with gr.Row():
                            undo = gr.Button('마지막 점 취소')
                            reset = gr.Button('편집 초기화')
                            save = gr.Button('저장')
                            cancel = gr.Button('취소')
                        enabled = gr.Checkbox(value=True, label='구역 감시 활성화')
                        person = gr.Checkbox(label='사람이 ROI 안에 있으면 경고')
                        reviewed = gr.Checkbox(label='운영자가 ROI 경계를 검토함')
                        exclude_reviewed = gr.Checkbox(label='운영자가 탐지 제외 구역을 검토함')
                        roi_msg = gr.Markdown('감시 활성과 화면의 ROI 표시는 따로 둡니다. 제외 구역의 기본값은 없습니다.')
                with gr.Column(visible=False) as page_alert:
                    with gr.Row():
                        alert_cam = gr.Dropdown(cameras, value='전체', label='카메라')
                        alert_sev = gr.Dropdown(['전체', '경고', '주의', '참고'], value='전체', label='심각도')
                        alert_ack = gr.Radio(['미확인', '확인함', '전체'], value='미확인', label='확인 상태')
                    alert_table = gr.Dataframe(
                        headers=alert_headers, datatype=alert_types, interactive=True,
                        static_columns=[1, 2, 3, 4, 5, 6], column_count=7, label='알림')
                    alert_msg = gr.Markdown('확인은 이 표의 체크박스로 저장됩니다.')
                    alert_detail = gr.Textbox(label='사건 상세', lines=10)
                    alert_image = gr.Image(label='증거', height=360, interactive=False, buttons=['fullscreen'])
                    review_label = gr.Radio(['실제 위험', '오탐', '판단 보류'], value='판단 보류', label='검토')
                    review_note = gr.Textbox(label='메모')
                    review_btn = gr.Button('검토 저장')
                with gr.Column(visible=False) as page_log:
                    with gr.Row():
                        log_from = gr.Textbox(label='시작 날짜 YYYY-MM-DD')
                        log_to = gr.Textbox(label='끝 날짜 YYYY-MM-DD')
                        log_cam = gr.Dropdown(cameras, value='전체', label='카메라')
                    with gr.Row():
                        log_sev = gr.Dropdown(['전체', '경고', '주의', '참고'], value='전체', label='심각도')
                        log_kind = gr.Dropdown(['전체', 'ZONE_EXIT_DANGER', 'OUTSIDE_OBSERVED', 'PERSON_INTRUSION'],
                                               value='전체', label='사건 종류')
                        log_ack = gr.Radio(['미확인', '확인함', '전체'], value='전체', label='확인')
                        log_view = gr.Radio(['기본 목록', '보관된 로그'], value='기본 목록', label='목록')
                    log_table = gr.Dataframe(headers=headers, interactive=False, label='이벤트 로그')
                    log_stat = gr.Markdown()
                    log_detail = gr.Textbox(label='선택 사건', lines=8)
                    with gr.Row():
                        log_reset = gr.Button('필터 초기화')
                        log_preview = gr.Button('로그 목록 비우기 대상 보기')
                        log_go = gr.Button('보관 처리')
                        log_cancel = gr.Button('보관 취소')
                        log_restore = gr.Button('복원')
                    log_ids = gr.State([])
                    log_msg = gr.Markdown('종료됐고 확인된 사건만 보관합니다. 진행 중·미확인 사건은 빠집니다.')
                with gr.Column(visible=False) as page_shot:
                    with gr.Row():
                        shot_from = gr.Textbox(label='시작 날짜')
                        shot_to = gr.Textbox(label='끝 날짜')
                        shot_cam = gr.Dropdown(cameras, value='전체', label='카메라')
                        shot_kind = gr.Dropdown(['전체', 'ZONE_EXIT_DANGER', 'OUTSIDE_OBSERVED', 'PERSON_INTRUSION'],
                                                value='전체', label='사건 종류')
                    gallery = gr.Gallery(label='스냅샷', columns=4, height=240, allow_preview=False)
                    shot_image = gr.Image(label='큰 미리보기', height=600, interactive=False, buttons=['fullscreen'])
                    shot_text = gr.Markdown()
                    shot_index = gr.State(0)
                    with gr.Row():
                        prev_btn = gr.Button('이전')
                        next_btn = gr.Button('다음')
                        shot_preview = gr.Button('스냅샷 목록 비우기 대상 보기')
                        shot_go = gr.Button('스냅샷 보관')
                        shot_restore = gr.Button('스냅샷 복원')
                    shot_ids = gr.State([])
                    shot_msg = gr.Markdown('보관은 갤러리 기본 목록만 바꿉니다. 사진 파일은 지우지 않습니다.')
                with gr.Column(visible=False) as page_ai:
                    ai_html = gr.HTML()
                    with gr.Accordion('원본 JSON', open=False):
                        ai_request = gr.JSON(label='보낸 내용')
                        ai_response = gr.JSON(label='받은 답')
                with gr.Column(visible=False) as page_set:
                    density = gr.Radio(['보통', '좁게'], value='보통', label='표시 밀도')
                    gr.Markdown('탐지 기준은 시연용입니다. 단위는 신뢰도 0~1, 시간 초, 완충 폭은 프레임 짧은 변 대비 비율입니다.')
                    set_cam = gr.Dropdown(codes, value=codes[0], label='적용 카메라')
                    set_class = gr.Textbox(label='클래스', value='giraffe')
                    candidate = gr.Number(label='후보 채택 점수', value=0.10)
                    confirm = gr.Number(label='경고 확정 점수', value=0.25)
                    samples = gr.Number(label='최소 관측 수', value=2, precision=0)
                    span = gr.Number(label='최소 확인 시간(초)', value=1.0)
                    gap = gr.Number(label='허용 누락 시간(초)', value=3.0)
                    boundary = gr.Number(label='경계 완충 폭(짧은 변 비율)', value=0.02)
                    safe_return = gr.Number(label='안전 복귀 확인(초)', value=0.8)
                    set_save = gr.Button('이 카메라·클래스에 저장')
                    set_msg = gr.Markdown()
        selected = gr.State('')
        pages = [page_home, page_cam, page_alert, page_log, page_shot, page_ai, page_set]

        titles = {'home': '홈', 'cameras': '카메라', 'alerts': '알림', 'logs': '이벤트 로그',
                  'shots': '스냅샷', 'ai': 'AI 보조 판단', 'settings': '설정'}

        def open_page(name):
            order = ['home', 'cameras', 'alerts', 'logs', 'shots', 'ai', 'settings']
            shown = tuple(gr.update(visible=name == item) for item in order)
            return shown + (f'<div class="sz-title">{titles[name]}</div>',)

        nav = pages + [title_html]
        home_btn.click(lambda: open_page('home'), None, nav)
        cam_btn.click(lambda: open_page('cameras'), None, nav)
        alert_btn.click(lambda: open_page('alerts'), None, nav)
        log_btn.click(lambda: open_page('logs'), None, nav)
        shot_btn.click(lambda: open_page('shots'), None, nav)
        ai_btn.click(lambda: open_page('ai'), None, nav)
        set_btn.click(lambda: open_page('settings'), None, nav)
        home_detail.click(lambda: open_page('alerts'), None, nav)
        fold.click(None, None, None, js="() => { document.getElementById('sidebar').classList.toggle('collapsed'); }")
        theme.change(None, theme, None, js=THEME_JS)
        density.change(None, density, None, js=DENSITY_JS)
        demo.load(None, None, theme, js=THEME_LOAD_JS)

        def tick_tables(camera, severity, ack_name, day_from, day_to, log_camera, log_severity, log_kind,
                        log_ack_name, view, event_id, shot_day_from, shot_day_to, shot_camera, shot_kind, index):
            head, badge_text, provider_text, clock_text = summary()
            alerts = refresh_alerts(camera, severity, ack_name)
            logs, stat = refresh_logs(day_from, day_to, log_camera, log_severity, log_kind, log_ack_name, view)
            detail_text, image, _note = detail(event_id)
            gallery_value, shot, shot_caption, shot_at, _shot_id = shot_view(
                shot_day_from, shot_day_to, shot_camera, shot_kind, index)
            story, request, response = ai_page(event_id)
            return (head, badge_text, provider_text, clock_text, home_alerts(), alerts, logs, stat,
                    detail_text, image, gallery_value, gallery_value, shot, shot_caption, story, request, response, shot_at)

        inputs = [alert_cam, alert_sev, alert_ack, log_from, log_to, log_cam, log_sev, log_kind, log_ack,
                  log_view, selected, shot_from, shot_to, shot_cam, shot_kind, shot_index]
        outputs = [counts, badge, provider, clock, home_table, alert_table, log_table, log_stat, alert_detail,
                   alert_image, gallery, home_gallery, shot_image, shot_text, ai_html, ai_request, ai_response, shot_index]
        demo.load(tick_tables, inputs, outputs)
        gr.Timer(2).tick(tick_tables, inputs, outputs)
        demo.load(video_tick, None, home_imgs + cam_imgs + [large])
        gr.Timer(0.2).tick(video_tick, None, home_imgs + cam_imgs + [large], queue=False)

        def use_row(evt: gr.SelectData):
            return selected_id(evt)

        home_table.select(lambda evt: selected_id(evt, 1), None, selected)
        alert_table.select(lambda evt: selected_id(evt, 1), None, selected)
        log_table.select(use_row, None, selected)
        home_table.change(apply_alert_checks, home_table, home_msg)
        alert_table.change(apply_alert_checks, alert_table, alert_msg)
        selected.change(detail, selected, [alert_detail, alert_image, review_note])
        selected.change(detail, selected, [log_detail, alert_image, review_note])
        review_btn.click(save_review, [selected, review_label, review_note], alert_msg)
        log_preview.click(lambda *args: preview_archive(*args, 'log'),
                          [log_from, log_to, log_cam, log_sev, log_kind, log_ack], [log_ids, log_msg])
        log_go.click(lambda ids: run_archive(ids, 'log'), log_ids, [log_msg, log_ids])
        log_cancel.click(lambda: ([], '보관을 취소했습니다. DB는 바꾸지 않았습니다.'), None, [log_ids, log_msg])
        log_restore.click(lambda event_id: run_restore(event_id, 'log'), selected, log_msg)
        log_reset.click(lambda: ('', '', '전체', '전체', '전체', '전체', '기본 목록'), None,
                        [log_from, log_to, log_cam, log_sev, log_kind, log_ack, log_view])
        shot_preview.click(lambda day_from, day_to, camera, kind: preview_archive(
                               day_from, day_to, camera, '전체', kind, '전체', 'snapshot'),
                           [shot_from, shot_to, shot_cam, shot_kind], [shot_ids, shot_msg])
        shot_go.click(lambda ids: run_archive(ids, 'snapshot'), shot_ids, [shot_msg, shot_ids])
        shot_restore.click(lambda event_id: run_restore(event_id, 'snapshot'), selected, shot_msg)
        prev_btn.click(lambda *args: move_shot(*args, -1), [shot_from, shot_to, shot_cam, shot_kind, shot_index], shot_index)
        next_btn.click(lambda *args: move_shot(*args, 1), [shot_from, shot_to, shot_cam, shot_kind, shot_index], shot_index)
        def apply_layout(value, code):
            single = value == '1분할'
            if single:
                set_focus(code)
            tiles = tuple(gr.update(visible=not single) for _ in cam_imgs)
            return tiles + (gr.update(visible=single, label=camera_label(code)),)

        cam_pick.change(show_one, cam_pick, [layout, *cam_imgs, large])
        layout.change(apply_layout, [layout, cam_pick], cam_imgs + [large])
        roi_visible.change(show_roi, roi_visible, cam_note)
        load_frame.click(roi_open, cam_pick, [board, safety_points, exclude_points, enabled, person, reviewed,
                                              exclude_reviewed, roi_msg])
        roi_click.__annotations__ = {'evt': gr.SelectData}
        board.select(roi_click, [safety_points, exclude_points, cam_pick, mode],
                     [board, safety_points, exclude_points, roi_msg])
        undo.click(roi_undo, [safety_points, exclude_points, cam_pick, mode],
                   [board, safety_points, exclude_points, roi_msg])
        reset.click(roi_open, cam_pick, [board, safety_points, exclude_points, enabled, person, reviewed,
                                         exclude_reviewed, roi_msg])
        cancel.click(roi_open, cam_pick, [board, safety_points, exclude_points, enabled, person, reviewed,
                                          exclude_reviewed, roi_msg])
        save.click(roi_save, [cam_pick, safety_points, exclude_points, enabled, person, reviewed, exclude_reviewed],
                   roi_msg)
        set_save.click(save_gate, [set_cam, set_class, candidate, confirm, samples, span, gap, boundary, safe_return],
                       set_msg)

        def jump(code):
            return (code, *show_one(code), *open_page('cameras'))

        for button, cam in zip(jump_buttons, original.CAMERAS):
            button.click(lambda code=cam['code']: jump(code), None,
                         [cam_pick, layout, *cam_imgs, large, *pages, title_html])
    return demo
