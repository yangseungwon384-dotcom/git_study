"""관제 사이트의 버전 관리: 승인된 버전을 보관하고, 후보 버전이 반려되면 되돌린다.

    python twin_release.py status            지금 운영 버전과, 작업 파일이 그 버전과 다른지
    python twin_release.py save v2 --note …   지금 작업 파일을 새 버전으로 보관하고 운영 버전으로 정한다 (승인)
    python twin_release.py restore v1        보관한 버전의 파일로 작업 파일을 되돌린다 (반려)
    python twin_release.py list              보관된 버전 목록
    python twin_release.py diff v1           그 버전과 지금 작업 파일이 다른 파일

코드 파일만 다룬다. 사건 DB, 계정, 증거 사진(twin_data/)은 버전과 상관없이 그대로 둔다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
RELEASES = HERE / 'releases'
STATE = RELEASES / 'state.json'
# 이 도구(twin_release.py) 자체는 버전에 넣지 않는다. 되돌려도 도구는 최신 그대로 남는다.
FILES = ('twin_server.py', 'twin_vision.py', 'twin_ai.py', 'twin_eval.py', 'twin_replay.py',
         'twin_control.html', 'docs/exit_alert_rule.md',
         'run_twin_control.bat', 'run_twin_control_network.bat', 'run_twin_control_tunnel.bat', 'run_ai_eval.bat')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def state():
    return json.loads(STATE.read_text(encoding='utf-8')) if STATE.is_file() else dict(live=None, history=[])


def write_state(data):
    RELEASES.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def changed(name):
    """보관된 버전과 지금 작업 파일이 다른 파일 이름들."""
    meta = json.loads((RELEASES / name / 'meta.json').read_text(encoding='utf-8'))
    return [f for f in FILES if digest(HERE / f) != meta['files'].get(f)]


def save(name, note):
    target = RELEASES / name
    if target.exists():
        raise SystemExit(f'{name}은 이미 있습니다. 다른 이름을 쓰세요.')
    target.mkdir(parents=True)
    files = {}
    for f in FILES:
        if (HERE / f).is_file():
            (target / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / f, target / f)
            files[f] = digest(HERE / f)
    meta = dict(name=name, note=note, saved_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=files)
    (target / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    data = state()
    data['history'].append(dict(name=name, note=note, at=meta['saved_at'], previous=data.get('live')))
    data['live'] = name
    write_state(data)
    print(f'{name} 보관, 운영 버전으로 지정 ({len(files)}개 파일)')


def restore(name):
    source = RELEASES / name
    if not (source / 'meta.json').is_file():
        raise SystemExit(f'{name} 버전이 없습니다.')
    meta = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
    for f in FILES:
        if f in meta['files']:
            (HERE / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / f, HERE / f)
        elif (HERE / f).is_file():
            # 그 버전에는 없던 파일: 지우지 않고 옆으로 치운다
            (HERE / f).rename(HERE / (f + '.removed'))
    data = state()
    data['history'].append(dict(name=name, note='되돌림', at=time.strftime('%Y-%m-%d %H:%M:%S'), previous=data.get('live')))
    data['live'] = name
    write_state(data)
    print(f'{name}으로 되돌렸습니다. 서버를 다시 시작해야 적용됩니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('list')
    p = sub.add_parser('save'); p.add_argument('name'); p.add_argument('--note', default='')
    p = sub.add_parser('restore'); p.add_argument('name')
    p = sub.add_parser('diff'); p.add_argument('name')
    args = parser.parse_args()
    if args.cmd == 'save':
        save(args.name, args.note)
    elif args.cmd == 'restore':
        restore(args.name)
    elif args.cmd == 'diff':
        print('\n'.join(changed(args.name)) or '같음')
    elif args.cmd == 'list':
        for entry in state()['history']:
            print(f"{entry['at']}  {entry['name']:<6} {entry['note']}")
    else:
        data = state()
        live = data.get('live')
        print('운영 버전:', live or '없음')
        if live:
            diff = changed(live)
            print('작업 파일:', '운영 버전과 같음' if not diff else '다름 (후보 작업 중): ' + ', '.join(diff))


if __name__ == '__main__':
    main()
