#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import urllib.request
import urllib.error


def read_meaningful_message(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().splitlines()
    except Exception:
        return ''
    # 过滤注释与空行
    meaningful = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        if line.strip() == '':
            continue
        meaningful.append(line)
    return '\n'.join(meaningful).strip()


def write_message(path: str, text: str) -> None:
    text = (text or '').strip()
    if not text:
        text = 'update'
    # 写入 UTF-8，避免中文乱码
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text + '\n')


def run_git(*args: str) -> str:
    try:
        cp = subprocess.run(['git', *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding='utf-8', errors='ignore', check=False)
        return cp.stdout.strip()
    except Exception:
        return ''


def collect_context() -> dict:
    files = run_git('diff', '--staged', '--name-status')
    stats = run_git('diff', '--staged', '--shortstat')
    numstat_lines = run_git('diff', '--staged', '--numstat').splitlines()
    numstat = '\n'.join(numstat_lines[:50])  # 限制规模，避免请求过大
    return {
        'files': files,
        'stats': stats,
        'numstat': numstat,
    }


def gen_with_google(context: dict, timeout_sec: int = 8):
    api_key = (
        os.getenv('GOOGLE_API_KEY')
        or os.getenv('GEMINI_API_KEY')
        or os.getenv('GOOGLEAI_API_KEY')
    )
    if not api_key:
        return None, 'no_api_key'

    url = (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        'gemini-1.5-flash-latest:generateContent?key=' + api_key
    )

    prompt = (
        '你是资深开发者，请基于以下变更生成一条简洁的提交信息'
        '（简体中文，50–72 字符，祈使语，一行，不要代码块/引号/前缀）:\n'
        '文件与状态:\n' + (context.get('files') or '(无)') + '\n\n'
        '统计:\n' + (context.get('stats') or '(无)') + '\n\n'
        '变更规模(新增\t删除\t文件):\n' + (context.get('numstat') or '(无)') + '\n\n'
        '只输出提交标题一行。'
    )

    body = {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': prompt}],
            }
        ],
        'generationConfig': {
            'temperature': 0.3,
            'topP': 0.9,
            'maxOutputTokens': 64,
        },
    }

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
    except Exception as e:
        return None, str(e)

    text = ''
    try:
        text = payload['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        # 尝试拼接候选文本
        try:
            for c in payload.get('candidates', []):
                for p in c.get('content', {}).get('parts', []):
                    t = p.get('text')
                    if t:
                        text += t
        except Exception:
            pass

    if not text:
        return None, 'empty_response'

    text = text.strip().splitlines()[0].strip()
    if len(text) > 100:
        text = text[:100]
    return text, None


def main():
    msg_file = sys.argv[1]
    source_type = sys.argv[2] if len(sys.argv) >= 3 else ''

    # 合并/压缩提交不改写
    if source_type in {'merge', 'squash'}:
        return 0

    current = read_meaningful_message(msg_file)
    if current and current.strip().lower() != 'update':
        # 用户已填写非 update 的信息，尊重用户
        return 0

    context = collect_context()
    text, err = gen_with_google(context)
    if text:
        write_message(msg_file, text)
    else:
        # 失败时保持为 update/空，避免阻塞提交
        if not current:
            write_message(msg_file, 'update')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except Exception:
        # 兜底：任何异常不影响提交
        sys.exit(0)

