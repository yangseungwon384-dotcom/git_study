"""AI 판단 채점 도구: 사람이 판정한 사건에 대해 TEV의 판단이 얼마나 맞는지 숫자로 낸다.

    python twin_eval.py                  저장된 AI 답으로 채점 (모델을 부르지 않는다)
    python twin_eval.py --ask            판정된 사건을 지금 규칙으로 TEV에 다시 묻고 채점 (Ollama 필요)
    python twin_eval.py --day 2026-10-12 그날 사건만
    python twin_eval.py --out 결과.csv   사건별 결과를 CSV로 저장

사람 판정은 관제 화면의 CSV 기록 탭에서 '위험 / 오탐 / 보류'로 남긴 것이다.
--ask 로 받은 답은 status가 eval_completed인 기록으로 남고, 관제 화면의 AI 참고 판단에는 나오지 않는다.
서버가 켜져 있어도 쓸 수 있다. 이 PC는 메모리가 작으니 --ask 는 서버를 끄고 돌리는 편이 빠르다.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
import time
import types
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'zoo_safe'))

import twin_ai  # noqa: E402


def load(db, day):
    """사건과, 사건별로 가장 나중에 받은 AI 답."""
    where, values = (' WHERE day=?', (day,)) if day else ('', ())
    events = []
    for row in db.execute('SELECT id, day, active, ended_at, end_reason, row_json, acknowledged_at, '
                          f'review_label, review_note FROM events{where} ORDER BY day, rowid', values):
        body = json.loads(row['row_json'])
        body.update(id=row['id'], day=row['day'], active=bool(row['active']), ended_at=row['ended_at'],
                    end_reason=row['end_reason'], acknowledged_at=row['acknowledged_at'],
                    review_label=row['review_label'], review_note=row['review_note'])
        events.append(body)
    answers = {}
    marks = ','.join('?' * len(twin_ai.ANSWERED))
    for row in db.execute(f'SELECT event_id, result_json, model_requested, status FROM decisions '
                          f'WHERE status IN ({marks}) AND result_json IS NOT NULL ORDER BY rowid',
                          twin_ai.ANSWERED):
        answers[row['event_id']] = dict(twin_ai.answer_of(json.loads(row['result_json'])),
                                        model=row['model_requested'], status=row['status'])
    return events, answers


def ask_all(db, labeled, events, model, url, timeout):
    """판정된 사건을 TEV에 다시 묻는다. 답은 eval_completed로 저장하고 돌려준다."""
    import zoo_vision_jev_new as zj
    args = types.SimpleNamespace(provider='ollama', jev_model=model, ollama_url=url, jev_timeout=timeout,
                                 score=False)
    by_day = {}
    for event in events:
        by_day.setdefault(event['day'], []).append(event)
    answers, failures = {}, 0
    for index, event in enumerate(labeled, 1):
        state = twin_ai.review_state(db, zj, event, by_day[event['day']])
        started = time.monotonic()
        try:
            result = zj.call_jev(state, args)
        except Exception as exc:
            failures += 1
            print(f'  {index}/{len(labeled)} 실패: {type(exc).__name__}')
            if failures >= 3 and not answers:
                raise SystemExit('TEV 응답을 받지 못했습니다. Ollama가 켜져 있고 모델이 설치돼 있는지 확인하세요.')
            continue
        result['latency_ms'] = round((time.monotonic() - started) * 1000)
        with db:
            db.execute('INSERT INTO decisions(id, event_id, requested_at, status, state_json, result_json, '
                       'error, model_requested, question_version, questions_json) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (uuid.uuid4().hex, event['id'], zj.now().isoformat(), 'eval_completed',
                        json.dumps(state, ensure_ascii=False), json.dumps(result), None, model,
                        zj.QUESTION_VERSION, json.dumps(zj.questions(False))))
        answers[event['id']] = dict(twin_ai.answer_of(result), model=model, status='eval_completed')
        print(f'  {index}/{len(labeled)} {result["latency_ms"]} ms')
    return answers


def percent(value):
    return '해당 없음' if value is None else f'{value * 100:.0f}%'


def report(result, labeled_count):
    lines = [f'사람이 판정한 사건 {labeled_count}건 중 AI 답이 있는 {result["total"]}건을 채점했습니다.']
    if not result['total']:
        return lines + ['채점할 사건이 없습니다. CSV 기록 탭에서 사건을 판정하고, AI 답이 없으면 --ask 를 붙이세요.']
    lines += ['', '사람 판정(세로) × AI 판단(가로)',
              '              ' + ''.join(f'{label:>10}' for label in twin_ai.HUMAN_LABELS)]
    for human in twin_ai.HUMAN_LABELS:
        lines.append(f'  {human:<10}' + ''.join(f'{result["table"][human][ai]:>12}' for ai in twin_ai.HUMAN_LABELS))
    means = result['needs_review_mean']
    auto = result['auto']
    lines += [
        '',
        f'전체 일치율                    {percent(result["agreement"])}',
        f'실제 위험을 AI도 위험으로 봄    {percent(result["danger_caught"])} (실제 위험 {result["danger_total"]}건)',
        f'실제 위험을 AI가 오탐으로 봄    {result["danger_missed"]}건  <- 가장 위험한 실수. 0이어야 한다',
        f'오탐을 AI도 오탐으로 봄         {percent(result["false_cleared"])} (오탐 {result["false_total"]}건)',
        '즉시 확인 필요도 평균           ' + ', '.join(
            f'{label} {"-" if means[label] is None else format(means[label], ".2f")}'
            for label in twin_ai.HUMAN_LABELS),
        '',
        f'자동 확인을 켰다면 (AI가 오탐, 확률 {auto["probability"]} 이상, 필요도 {auto["needs_review"]} 이하)',
        f'  자동 확인됐을 사건 {auto["count"]}건, 그중 실제 위험 {auto["wrong"]}건',
    ]
    if result['total'] < 30:
        lines += ['', f'주의: {result["total"]}건은 결론을 내기에 적습니다. 종류별로 고르게 200건쯤 판정한 뒤 다시 보세요.']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'twin_data')
    parser.add_argument('--day', help='YYYY-MM-DD. 비우면 전체')
    parser.add_argument('--ask', action='store_true', help='판정된 사건을 TEV에 다시 묻는다')
    parser.add_argument('--model', default='tev1:0.8b')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--auto-probability', type=float, default=0.6)
    parser.add_argument('--auto-needs-review', type=float, default=0.3)
    parser.add_argument('--out', type=Path, help='사건별 결과를 저장할 CSV 경로')
    args = parser.parse_args()

    path = args.data_dir / 'events.sqlite3'
    if not path.is_file():
        raise SystemExit(f'사건 DB가 없습니다: {path}')
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        events, answers = load(db, args.day)
        labeled = [event for event in events if event['review_label'] in twin_ai.HUMAN_LABELS]
        if args.ask and labeled:
            print(f'{len(labeled)}건을 {args.model}에 묻습니다.')
            answers.update(ask_all(db, labeled, events, args.model, args.ollama_url, args.timeout))
    finally:
        db.close()

    scored = [(event, answers[event['id']]) for event in labeled if event['id'] in answers]
    result = twin_ai.score([(event['review_label'], answer) for event, answer in scored],
                           args.auto_probability, args.auto_needs_review)
    print()
    print('\n'.join(report(result, len(labeled))))

    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['detected_at', 'camera_id', 'object', 'event_type', 'severity', 'human_label',
                             'ai_label', 'ai_choice', 'ai_probability', 'ai_needs_review', 'match', 'ai_model'])
            for event, answer in scored:
                ai = twin_ai.predicted_label(answer['choice'])
                writer.writerow([event.get('detected_at'), event.get('camera_id'), event.get('object'),
                                 event.get('event_type'), event.get('severity'), event['review_label'], ai,
                                 answer['choice'], round(answer['probability'], 3),
                                 round(answer['needs_review'], 3), int(ai == event['review_label']),
                                 answer['model']])
        print(f'\n사건별 결과 저장: {args.out}')


if __name__ == '__main__':
    main()
