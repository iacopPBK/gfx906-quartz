"""Per-architecture chat dialects: how a model's generated text splits into
reasoning, content and tool calls, whole or streamed, keyed on the GGUF
general.architecture."""
import json
import re
import uuid


def _call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}}


def _param_types(tools):
    """JSON schema type per function name and parameter, from the request's tools."""
    types = {}
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        props = ((f.get("parameters") or {}).get("properties") or {}) if isinstance(f, dict) else {}
        types[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    return types


class Dialect:
    """The interface. A stream gets the callbacks reasoning(s), content(s) and
    calls(list), each returning bool, where False means the client is gone."""
    special_pieces = False      # decode control tokens into the text (the parser reads them)

    def think_open(self, rendered):
        """Whether the rendered prompt leaves the model inside its reasoning."""
        return False

    def answer_open(self, rendered):
        """The rendered prompt with thinking off: the model's turn opened as the answer.
        Templates that skip thinking on their own (enable_thinking) leave it as is."""
        return rendered

    def parse(self, text, think_open, tools):
        """Split a whole answer into (reasoning or None, content, tool calls)."""
        raise NotImplementedError

    def stream(self, think_open, tools, reasoning, content, calls):
        """The streaming parser. feed(s) -> bool, close() -> the tool calls sent."""
        raise NotImplementedError

    def flags(self, text, think_open, content, calls):
        """Trace flags for answers that look cut short or malformed."""
        return []


class Qwen(Dialect):
    """Reasoning sits between the <think> and </think> tags the template opens, and
    tool calls come in XML or JSON <tool_call> blocks."""
    TOOL_OPEN = "<tool_call>"

    def think_open(self, rendered):
        return rendered.rstrip(" ").endswith("<think>\n")

    @staticmethod
    def split_reasoning(text, think_open):
        """With thinking open, the text up to </think> is reasoning, all of it if the
        tag never comes."""
        if not think_open:
            return None, text
        if "</think>" in text:
            r, c = text.split("</think>", 1)
            return r.replace("<think>", "").strip(), c.lstrip("\n")
        return text.replace("<think>", "").strip(), ""

    @classmethod
    def parse_tool_calls(cls, text, tools):
        """Split XML or JSON tool calls off the text, giving the content before the first
        call and OpenAI tool_calls; values parse as JSON unless the schema says string."""
        i = text.find(cls.TOOL_OPEN)
        if i < 0:
            return text, []
        content = text[:i].rstrip()
        types = _param_types(tools)
        calls = []
        # </tool_call> closes the element only when the next <tool_call> or the end
        # follows, so values that contain the tag itself (code, docs) survive
        for blk in re.finditer(r"<tool_call>(.*?)(?:</tool_call>\s*(?=<tool_call>|$)|$)", text[i:], re.S):
            body = blk.group(1)
            name, args = None, {}
            fm = (re.search(r"<function=([^>\n]+)>(.*)</function>\s*$", body, re.S)
                  or re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", body, re.S))
            if fm:
                name = fm.group(1).strip()
                pt = types.get(name, {})
                for pm in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>\s*(?=<parameter=[^>\n]+>|$)",
                                      fm.group(2), re.S):
                    k, v = pm.group(1).strip(), pm.group(2)
                    if v.startswith("\n"):
                        v = v[1:]
                    if v.endswith("\n"):
                        v = v[:-1]
                    if pt.get(k) != "string":
                        try:
                            v = json.loads(v)
                        except ValueError:
                            pass
                    args[k] = v
            elif body.strip().startswith("{"):
                try:
                    j = json.loads(body.strip())
                    name, args = j.get("name"), j.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                except ValueError:
                    name = None
            if name:
                calls.append(_call(name, args))
        return content, calls

    def parse(self, text, think_open, tools):
        reasoning, content = self.split_reasoning(text, think_open)
        calls = []
        if tools:
            content, calls = self.parse_tool_calls(content, tools)
        return reasoning, content, calls

    def stream(self, think_open, tools, reasoning, content, calls):
        return _QwenStream(self, think_open, tools, reasoning, content, calls)

    def flags(self, text, think_open, content, calls):
        f = []
        if think_open and "</think>" not in text:
            f.append("think never closed")
        if not calls and ("<tool_call" in (content or "") or "<function=" in (content or "")):
            f.append("unparsed tool call text")
        return f


class _QwenStream:
    def __init__(self, d, think_open, tools, reasoning, content, calls):
        self.d, self.tools = d, tools
        self.send_r, self.send_c, self.send_calls = reasoning, content, calls
        self.think, self.buf = think_open, ""                # reasoning until </think>
        self.on, self.pend, self.tail = False, "", ""        # tools: content up to <tool_call> is sent, the rest held

    def _content(self, s):
        if not self.tools:
            return self.send_c(s)
        if self.on:
            self.tail += s
            return True
        T = self.d.TOOL_OPEN
        buf = self.pend + s
        i = buf.find(T)
        if i >= 0:
            self.on, self.pend, self.tail = True, "", buf[i:]
            out = buf[:i]
        else:                                    # hold a suffix that may open a call
            k = next((n for n in range(min(len(T) - 1, len(buf)), 0, -1) if T.startswith(buf[-n:])), 0)
            out, self.pend = (buf[:-k], buf[-k:]) if k else (buf, "")
        return self.send_c(out) if out else True

    def feed(self, s):
        if not self.think:
            return self._content(s)
        self.buf += s
        if "</think>" in self.buf:
            r, c = self.buf.split("</think>", 1)
            self.think = False
            ok = True
            r = r.replace("<think>", "")
            if r:
                ok = self.send_r(r)
            c = c.lstrip("\n")
            if c and ok:
                ok = self._content(c)
            return ok
        keep = len("</think>") - 1              # a split closing tag stays buffered
        out, self.buf = self.buf[:-keep] if len(self.buf) > keep else "", \
            self.buf[-keep:] if len(self.buf) > keep else self.buf
        out = out.replace("<think>", "")
        return self.send_r(out) if out else True

    def close(self):
        if self.think and self.buf:
            self.send_r(self.buf.replace("<think>", ""))
        if self.tools and self.pend and not self.on:
            self.send_c(self.pend)
        calls = []
        if self.tools and self.on:
            _, calls = self.d.parse_tool_calls(self.tail, self.tools)
            if calls:
                self.send_calls(calls)
            else:                                # no parsable call: the text goes out as content
                self.send_c(self.tail)
        return calls


class Inkling(Dialect):
    """Model messages are control-token blocks:
    <|message_model|>[name]<|content_KIND|>body<|end_message|>, with KIND one of
    thinking, text, invoke_tool_json ({"name", "args"}) or invoke_tool_text. The
    answer ends with <|content_model_end_sampling|>."""
    special_pieces = True
    _KIND = {"<|content_thinking|>": "think", "<|content_text|>": "text",
             "<|content_invoke_tool_json|>": "tool_json", "<|content_invoke_tool_text|>": "tool_text"}
    _ENDS = ("<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>")
    _MARKS = list(_KIND) + list(_ENDS)
    _TOK = re.compile("|".join(re.escape(m) for m in _MARKS))   # other <|..|> text stays text
    _HOLD = max(len(m) for m in _MARKS) - 1

    def answer_open(self, rendered):
        # the template has no generation prompt: the model writes <|message_model|> itself and
        # then picks thinking or text, thinking even at effort 0. Thinking off opens the turn as text.
        if not rendered.endswith("<|message_model|>"):
            rendered += "<|message_model|>"
        return rendered + "<|content_text|>"

    def parse(self, text, think_open, tools):
        out = {"r": [], "c": [], "calls": []}
        s = _InklingStream(lambda x: out["r"].append(x) or True, lambda x: out["c"].append(x) or True,
                           lambda x: out["calls"].extend(x) or True)
        s.feed(text)
        s.close()
        r = "".join(out["r"]).strip()
        return (r if out["r"] else None), "".join(out["c"]).strip("\n"), out["calls"]

    def stream(self, think_open, tools, reasoning, content, calls):
        return _InklingStream(reasoning, content, calls)

    def flags(self, text, think_open, content, calls):
        f = []
        if "<|message_model|>" in text and "<|end_message|>" not in text.rsplit("<|message_model|>", 1)[1]:
            f.append("message never closed")
        return f


class _InklingStream:
    def __init__(self, reasoning, content, calls):
        self.send_r, self.send_c, self.send_calls = reasoning, content, calls
        self.mode, self.name, self.body, self.buf = "out", "", "", ""   # out | head | think | text | tool_* | skip
        self.calls = []

    def _text(self, s):
        if not s:
            return True
        if self.mode == "think":
            return self.send_r(s)
        if self.mode in ("text", "out"):
            return self.send_c(s)
        if self.mode == "head":
            self.name += s
        elif self.mode.startswith("tool"):
            self.body += s
        return True

    def _end_tool(self):
        name, body = self.name.strip(), self.body.strip()
        if self.mode == "tool_json":
            try:
                j = json.loads(body)
                self.calls.append(_call(j.get("name") or name, j.get("args") or j.get("arguments") or {}))
                return
            except (ValueError, AttributeError):
                pass
        if name:
            self.calls.append(_call(name, body))

    def _marker(self, m):
        if m == "<|message_model|>":
            if self.mode.startswith("tool"):
                self._end_tool()
            self.mode, self.name, self.body = "head", "", ""
        elif m in Inkling._KIND:
            self.mode = Inkling._KIND[m]
        elif m == "<|end_message|>":
            if self.mode.startswith("tool"):
                self._end_tool()
            self.mode = "skip"
        elif m == "<|content_model_end_sampling|>":
            self.mode = "skip"

    def feed(self, s):
        self.buf += s
        ok = True
        while ok:
            m = Inkling._TOK.search(self.buf)
            if not m:
                break
            ok = self._text(self.buf[:m.start()])
            self._marker(m.group(0))
            self.buf = self.buf[m.end():]
        # a marker split across deltas stays buffered
        i = self.buf.rfind("<", max(0, len(self.buf) - Inkling._HOLD))
        hold = i >= 0 and any(m.startswith(self.buf[i:]) for m in Inkling._MARKS)
        out, self.buf = (self.buf[:i], self.buf[i:]) if hold else (self.buf, "")
        return self._text(out) and ok

    def close(self):
        self._text(self.buf)
        self.buf = ""
        if self.mode.startswith("tool"):
            self._end_tool()
            self.mode = "skip"
        if self.calls:
            self.send_calls(self.calls)
        return self.calls


DIALECTS = {"qwen35": Qwen, "inkling2_loop": Inkling}


def for_arch(arch):
    if arch not in DIALECTS:
        raise RuntimeError("no chat dialect for architecture '%s' (quartz/server/dialects.py)" % arch)
    return DIALECTS[arch]()
