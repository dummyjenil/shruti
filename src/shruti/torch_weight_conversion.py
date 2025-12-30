import re
import ast
import operator

# safe arithmetic evaluator (allows + - * / // % and parentheses)
_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def eval_math_expr(expr: str) -> int:
    """
    Safely evaluate a small arithmetic expression and return int(result).
    Allows + - * / // % and parentheses. No names, no function calls.
    """
    node = ast.parse(expr, mode='eval').body

    def _eval(n):
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = type(n.op)
            if op in _ops:
                return _ops[op](left, right)
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            op = type(n.op)
            if op in _ops:
                return _ops[op](operand)
        raise ValueError(f"Unsupported expression: {expr!r}")

    return int(_eval(node))


def parse_rules(changes: str):
    rules = [tuple(line.split(",")) for line in changes.strip().splitlines()]
    indexed = [(a, b) for a, b in rules if "%d" in a]
    simple = [(a, b) for a, b in rules if "%d" not in a]
    return indexed, simple


def apply_indexed_rule_to_key(key: str, src: str, dst: str):
    """
    Build a regex from src (which contains one or more %d)
    and replace all matches in `key` using dst where %d placeholders
    are substituted by captured integers. Parenthesized math in dst
    like ( %d/2 + 0.5 ) are evaluated and replaced with int().
    """
    parts = src.split("%d")
    # build regex by escaping parts and inserting numeric capture groups
    regex = "".join(re.escape(p) + (r"(\d+)" if i < len(parts) - 1 else "") for i, p in enumerate(parts))

    def repl(m):
        groups = list(m.groups())
        out = dst
        # sequentially replace each %d in dst with corresponding captured number
        for g in groups:
            out = out.replace("%d", str(int(g)), 1)

        # evaluate parenthesized arithmetic expressions and replace them with int result
        def _re_eval(parm):
            expr = parm.group(1)
            return str(eval_math_expr(expr))

        out = re.sub(r"\((.*?)\)", _re_eval, out)
        return out

    new_key, count = re.subn(regex, repl, key)
    return new_key, (count > 0)


def torch_weight_conversion(state_dict: dict, changes: str) -> dict:
    """
    Uses the exact `changes` multiline string you provided.
    - Indexed rules (with %d) are applied first (they are regex-based).
    - Then simple substring replacements are applied.
    Returns a new state_dict with renamed keys (values preserved).
    """
    indexed_rules, simple_rules = parse_rules(changes)
    new_sd = {}

    for key, val in state_dict.items():
        new_key = key

        # apply all indexed rules (they will replace matched substrings)
        for src, dst in indexed_rules:
            new_key, applied = apply_indexed_rule_to_key(new_key, src, dst)

        # then apply simple replacements
        for src, dst in simple_rules:
            if src in new_key:
                new_key = new_key.replace(src, dst)

        new_sd[new_key] = val

    return new_sd
