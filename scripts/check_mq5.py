#!/usr/bin/env python3
"""MQL5 源码括号配平检查(正确处理 // 注释、块注释、"字符串"、'字符'与转义)。"""
import sys

def check(path):
    src = open(path, encoding='utf-8-sig').read()
    i, n, line = 0, len(src), 1
    state = 'code'
    stack, errors = [], []
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ''
        if ch == '\n':
            line += 1
            if state == 'line_comment':
                state = 'code'
            i += 1
            continue
        if state == 'code':
            if ch == '"':
                state = 'str'
            elif ch == "'":
                state = 'chr'
            elif ch == '/' and nxt == '/':
                state = 'line_comment'; i += 2; continue
            elif ch == '/' and nxt == '*':
                state = 'block_comment'; i += 2; continue
            elif ch in '([{':
                stack.append((ch, line))
            elif ch in ')]}':
                if not stack:
                    errors.append(f'line {line}: unmatched {ch}')
                else:
                    o, ol = stack.pop()
                    pair = {'}': '{', ')': '(', ']': '['}[ch]
                    if o != pair:
                        errors.append(f'line {line}: {ch} closes {o} (opened line {ol})')
        elif state == 'str':
            if ch == '\\':
                i += 2; continue
            if ch == '"':
                state = 'code'
        elif state == 'chr':
            if ch == '\\':
                i += 2; continue
            if ch == "'":
                state = 'code'
        elif state == 'block_comment':
            if ch == '*' and nxt == '/':
                state = 'code'; i += 2; continue
        i += 1
    if state != 'code':
        errors.append(f'EOF in state: {state}')
    for o, ol in stack:
        errors.append(f'unclosed {o} at line {ol}')
    print(f'{path}: {len(src.splitlines())} lines')
    if errors:
        print('FAIL')
        for e in errors[:20]:
            print(' ', e)
        return 1
    print('PASS: brackets balanced, strings/chars/comments all closed')
    return 0

if __name__ == '__main__':
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else
                   '/home/z/my-project/download/xauusd_ml_v2/mt5_package/XauV3BalEnsEA.mq5'))
