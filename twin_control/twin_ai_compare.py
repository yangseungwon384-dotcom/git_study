"""판단 AI 비교 (R8): 같은 기린 사건 묶음을 모델·입력별로 보내 정답률과 확률, 응답 시간, 메모리를 잰다.

    python twin_ai_compare.py --data-dir <사건 DB 폴더> --models tev1:0.8b tev1:4b-q4_K_M
    python twin_ai_compare.py ... --set-out docs/giraffe_eval_set.csv     평가 사건 묶음과 정답을 저장한다

정답 기준 (docs/ai_rule.md의 R8 절):
  기린 사건: 기린은 영상 13.7초에 펜스를 넘어 영상 끝까지 구역 밖에 있다 → 13.7초 이후 BOUNDARY_CONCERN
  코끼리 오탐 (v2 이전, 2026-10-15 10:07 전 요청): 코끼리는 영상 내내 울타리 안이었다 (R1 분석) → NORMAL
입력 'base'는 엔진 원래 요약, 'geometry'는 v9 후보가 덧붙인 evidence.geometry를 포함한 요약이다.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'zoo_safe'))

CROSS_SEC = 13.7
CAMERA = 'ZOO_CAM_GIRAFFE_01'


def label(state):
    if state['source']['camera_id'] != CAMERA or state['subject']['detected_class'] == 'person':
        return 'NORMAL'
    return 'BOUNDARY_CONCERN' if state['source']['video_position_sec'] >= CROSS_SEC else 'NORMAL'


def eval_set(data_dir, size, since, negatives):
    """기린 이탈: geometry가 들어간 기린 요청(since 이후)을 영상 위치가 고르게 퍼지도록 고른다.
    평상시: v2 이전 코끼리 오탐 요청(코끼리는 실제로 울타리 안, geometry 없음)을 negatives건 고른다."""
    db = sqlite3.connect(f'file:{data_dir / "events.sqlite3"}?mode=ro', uri=True)
    pick = lambda xs, n: [xs[round(i * (len(xs) - 1) / max(n - 1, 1))] for i in range(min(n, len(xs)))]
    giraffe = [(rid, json.loads(js)) for rid, js in db.execute(
        'SELECT d.id, d.state_json FROM decisions d JOIN events e ON e.id = d.event_id '
        'WHERE e.camera = ? AND d.state_json IS NOT NULL AND d.requested_at >= ? ORDER BY d.rowid', (CAMERA, since))]
    giraffe = [x for x in giraffe if 'geometry' in x[1]['evidence'] and x[1]['subject']['detected_class'] != 'person']
    elephant = [(rid, json.loads(js)) for rid, js in db.execute(
        "SELECT d.id, d.state_json FROM decisions d JOIN events e ON e.id = d.event_id "
        "WHERE e.camera = 'ZOO_CAM_ELEPHANT_01' AND d.state_json IS NOT NULL "
        "AND d.requested_at >= '2026-10-15' AND d.requested_at < '2026-10-15T10:07' ORDER BY d.rowid")]
    elephant = [x for x in elephant if x[1]['subject']['detected_class'] != 'person']
    chosen = pick(sorted(giraffe, key=lambda x: x[1]['source']['video_position_sec']), size) + pick(elephant, negatives)
    return [dict(id=rid, state=st, label=label(st)) for rid, st in chosen]

def strip(state):
    out = copy.deepcopy(state)
    out['evidence'].pop('geometry', None)
    return out


def ask(url, model, state, timeout):
    import zoo_vision_jev_new as zj
    body = json.dumps({'model': model, 'state': state, 'questions': zj.questions(False)}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip('/') + '/v1/systemone', data=body, method='POST',
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        answer = json.loads(r.read())
    return answer['answers']['situation'], (time.monotonic() - started) * 1000


def loaded_size(url, model):
    with urllib.request.urlopen(url.rstrip('/') + '/api/ps', timeout=5) as r:
        for m in json.loads(r.read()).get('models', []):
            if m.get('name') == model:
                return m.get('size')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'twin_data')
    parser.add_argument('--since', default='', help='이 시각 이후 요청만 쓴다 (ISO)')
    parser.add_argument('--size', type=int, default=24, help='기린 사건 수')
    parser.add_argument('--negatives', type=int, default=10, help='평상시가 정답인 코끼리 오탐 사건 수')
    parser.add_argument('--models', nargs='+', default=['tev1:0.8b'])
    parser.add_argument('--inputs', nargs='+', default=['base', 'geometry'])
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--set-out', type=Path)
    parser.add_argument('--out', type=Path, help='사건별 결과 CSV')
    args = parser.parse_args()
    cases = eval_set(args.data_dir, args.size, args.since, args.negatives)
    if not cases:
        raise SystemExit('평가할 기린 사건 요청이 없습니다. v9 후보 서버를 --ollama로 몇 분 돌린 뒤 다시 실행하세요.')
    print(f'평가 사건 {len(cases)}건: ' + ', '.join(f"{k} {sum(c['label'] == k for c in cases)}" for k in ('BOUNDARY_CONCERN', 'NORMAL')), flush=True)
    if args.set_out:
        with open(args.set_out, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['request_id', 'camera', 'video_sec', 'object', 'roi_relation', 'previously_safe', 'foot_outside_distance_ratio',
                        'track_observed_sec', 'sec_since_last_seen_inside', 'label'])
            for c in cases:
                st, g = c['state'], c['state']['evidence'].get('geometry', {})
                w.writerow([c['id'], st['source']['camera_id'], st['source']['video_position_sec'], st['subject']['detected_class'],
                            st['evidence']['current_roi_relation'], st['evidence']['previously_observed_safe'],
                            g.get('foot_outside_distance_ratio'), g.get('track_observed_sec'),
                            g.get('sec_since_last_seen_inside'), c['label']])
    rows = []
    print(f"{'모델':18s} {'입력':9s} {'정답률':>6s} {'이탈맞힘':>8s} {'평상시맞힘':>9s} {'정답확률':>8s} {'이탈확률(이탈)':>12s} {'이탈확률(평상시)':>14s} {'1순위확률':>9s} {'응답중앙':>8s} {'메모리':>8s}", flush=True)
    for model in args.models:
        ask(args.ollama_url, model, strip(cases[0]['state']), args.timeout)   # 모델 올리기 (측정에서 뺌)
        size = loaded_size(args.ollama_url, model)
        for kind in args.inputs:
            res = []
            for c in cases:
                st = c['state'] if kind == 'geometry' else strip(c['state'])
                a, ms = ask(args.ollama_url, model, st, args.timeout)
                res.append((c, a, ms))
                rows.append(dict(model=model, input=kind, request_id=c['id'], label=c['label'], choice=a['choice'],
                                 p_label=round(a['probabilities'][c['label']], 3),
                                 p_boundary=round(a['probabilities']['BOUNDARY_CONCERN'], 3),
                                 p_normal=round(a['probabilities']['NORMAL'], 3), top=round(a['probabilities'][a['choice']], 3),
                                 latency_ms=round(ms)))
            acc = sum(a['choice'] == c['label'] for c, a, _ in res) / len(res)
            p_label = statistics.mean(a['probabilities'][c['label']] for c, a, _ in res)
            pos = [a['probabilities']['BOUNDARY_CONCERN'] for c, a, _ in res if c['label'] == 'BOUNDARY_CONCERN']
            top = statistics.mean(a['probabilities'][a['choice']] for _, a, _ in res)
            hit = lambda lab: [a['choice'] == lab for c, a, _ in res if c['label'] == lab]
            neg = [a['probabilities']['BOUNDARY_CONCERN'] for c, a, _ in res if c['label'] == 'NORMAL']
            lat = statistics.median(ms for *_, ms in res)
            b, n = hit('BOUNDARY_CONCERN'), hit('NORMAL')
            print(f"{model:18s} {kind:9s} {acc:6.0%} {sum(b):3d}/{len(b):<4d} {sum(n):4d}/{len(n):<4d} {p_label:8.2f} "
                  f"{statistics.mean(pos) if pos else 0:12.2f} {statistics.mean(neg) if neg else 0:14.2f} {top:9.2f} "
                  f"{lat / 1000:7.2f}s {(size or 0) / 2**30:7.2f}GB", flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
