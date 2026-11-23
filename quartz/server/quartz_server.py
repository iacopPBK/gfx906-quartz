#!/usr/bin/env python3
"""OpenAI-compatible server for the quartz engine (chat / completions, SSE, tools, images).
Run: python3 quartz/server/quartz_server.py --model <target.gguf> [--dspine <draft.gguf>] [--mmproj <projector.gguf>]"""
import argparse
import base64
import codecs
import ctypes
import json
import re
import os
import random
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dialects

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QzStats(ctypes.Structure):
    _fields_ = [("prefill_ms", ctypes.c_double), ("decode_ms", ctypes.c_double),
                ("n_prompt", ctypes.c_int), ("n_gen", ctypes.c_int),
                ("prefill_cycles", ctypes.c_int), ("decode_cycles", ctypes.c_int),
                ("decode_tok", ctypes.c_int), ("stop", ctypes.c_int), ("n_cached", ctypes.c_int),
                ("dspine", ctypes.c_int), ("wide_cycles", ctypes.c_int)]


class QzSampling(ctypes.Structure):
    _fields_ = [("temp", ctypes.c_float), ("top_k", ctypes.c_int), ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float), ("min_keep", ctypes.c_int), ("seed", ctypes.c_uint32),
                ("n_order", ctypes.c_int), ("order", ctypes.c_int * 8)]


class QzInfo(ctypes.Structure):
    _fields_ = [("arch", ctypes.c_char * 32), ("name", ctypes.c_char * 128), ("n_vocab", ctypes.c_int),
                ("max_ctx", ctypes.c_int), ("img_pad", ctypes.c_int), ("vision", ctypes.c_int),
                ("dspine", ctypes.c_int)]


EMIT_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)


class Engine:
    """libquartz + tokenizer; generate() streams decoded text pieces."""

    def __init__(self, lib, model, max_ctx, slot_t, slack):
        L = ctypes.CDLL(lib)
        L.qz_tok_open.restype = ctypes.c_void_p
        L.qz_tok_open.argtypes = [ctypes.c_char_p]
        L.qz_tok_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        L.qz_tok_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.qz_tok_eog.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        L.qz_tok_chat_template.restype = ctypes.c_char_p
        L.qz_tok_chat_template.argtypes = [ctypes.c_void_p]
        L.qz_create.restype = ctypes.c_void_p
        L.qz_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.qz_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int32), ctypes.c_int, EMIT_CB, ctypes.c_void_p,
                                  ctypes.POINTER(QzStats)]
        L.qz_generate_s.argtypes = L.qz_generate.argtypes + [ctypes.POINTER(QzSampling)]
        L.qz_abort.argtypes = [ctypes.c_void_p]
        L.qz_max_ctx.argtypes = [ctypes.c_void_p]
        L.qz_model_info_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(QzInfo)]
        L.qz_vision_attach.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.qz_image_add.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.qz_image_last_ms.restype = ctypes.c_double
        L.qz_image_last_ms.argtypes = [ctypes.c_void_p]
        L.qz_generate_mm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_int] + L.qz_generate_s.argtypes[3:]
        self.L = L
        self.vision = False
        self.tok = L.qz_tok_open(model.encode())
        if not self.tok:
            raise RuntimeError("tokenizer load failed")
        buf = (ctypes.c_int32 * 64)()
        n = L.qz_tok_eog(self.tok, buf, 64)
        self.eog = list(buf[:n])
        self.ctx = L.qz_create(model.encode(), max_ctx, slot_t, slack)
        if not self.ctx:
            raise RuntimeError("engine load failed")
        self.max_ctx = L.qz_max_ctx(self.ctx)
        self.info = QzInfo()
        L.qz_model_info_get(self.ctx, ctypes.byref(self.info))
        self.arch = self.info.arch.decode()
        self.dialect = dialects.for_arch(self.arch)
        self.img_pad = self.info.img_pad
        self.lock = threading.Lock()
        self._pb = ctypes.create_string_buffer(512)
        self.template = self._load_template()

    def drafter_name(self, drafted):
        """The log's decode mode: qwen35 drafts with DSpine or MTP, Inkling with DSpark or not at all."""
        if self.arch == "qwen35":
            return "dspine" if drafted else "mtp"
        return "dspark" if drafted else "plain"

    def _load_template(self):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        raw = self.L.qz_tok_chat_template(self.tok)
        if raw is None:
            return None
        src = raw.decode("utf-8")
        # HF's {% generation %} tags (unknown to jinja2) become an always-true if
        src = re.sub(r"\{%(-?)\s*generation\s*(-?)%\}", r"{%\1 if true \2%}", src)
        src = re.sub(r"\{%(-?)\s*endgeneration\s*(-?)%\}", r"{%\1 endif \2%}", src)

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False, separators=None: json.dumps(
            x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(src)

    def attach_vision(self, mmproj):
        if self.L.qz_vision_attach(self.ctx, mmproj.encode()) != 0:
            raise RuntimeError("projector load failed: %s" % mmproj)
        self.vision = True

    def add_images(self, blobs):
        """Encoded image files -> [(id, nx, ny)] (cached by content in libquartz)."""
        out = []
        with self.lock:
            for b in blobs:
                nx, ny = ctypes.c_int(), ctypes.c_int()
                t0 = time.time()
                i = self.L.qz_image_add(self.ctx, b, len(b), ctypes.byref(nx), ctypes.byref(ny))
                if i < 0:
                    raise ValueError("image %d could not be decoded or encoded" % len(out))
                log("image %d: %dx%d = %d tok, %.0f ms"
                    % (len(out), nx.value, ny.value, nx.value * ny.value, 1000 * (time.time() - t0)))
                out.append((i, nx.value, ny.value))
        return out

    def expand_images(self, prompt, imgs):
        """One <|image_pad|> per image (as the template writes it) -> nx * ny."""
        out, k = [], 0
        for t in prompt:
            if t == self.img_pad:
                if k >= len(imgs):
                    raise ValueError("more image tokens than images")
                out += [t] * (imgs[k][1] * imgs[k][2])
                k += 1
            else:
                out.append(t)
        if k != len(imgs):
            raise ValueError("%d images, %d image tokens in the prompt" % (len(imgs), k))
        return out

    def render(self, messages, tools=None, kwargs=None):
        if self.template is None:
            raise ValueError("model has no chat template")
        args = {"messages": messages, "add_generation_prompt": True}
        if tools:
            args["tools"] = tools
        args.update(kwargs or {})
        return self.template.render(**args)

    def tokenize(self, text, special=True):
        cap = len(text) + 16
        buf = (ctypes.c_int32 * cap)()
        n = self.L.qz_tok_encode(self.tok, text.encode("utf-8"), 0, 1 if special else 0, buf, cap)
        if n < 0:
            cap = -n
            buf = (ctypes.c_int32 * cap)()
            n = self.L.qz_tok_encode(self.tok, text.encode("utf-8"), 0, 1 if special else 0, buf, cap)
        return list(buf[:n])

    def tokenize_chat(self, text):
        """Tokenize a rendered prompt: control tokens only where the template wrote them."""
        out = []
        for i, seg in enumerate(_SPECIAL_RE.split(text)):
            if not seg:
                continue
            if i % 2:
                out += self.tokenize(seg, special=True)
            else:
                out += self.tokenize(seg.replace(_ESC, "<|"), special=False)
        return out

    def piece(self, t, special=False):
        n = self.L.qz_tok_piece(self.tok, t, self._pb, 512, 1 if special else 0)
        return self._pb.raw[:n] if n > 0 else b""

    def detok(self, ids, special=True):
        return b"".join(self.piece(t, special) for t in ids).decode("utf-8", "replace")

    def generate(self, prompt, max_new, on_text, stop_strings=(), want_ids=False, ignore_eos=False, sampling=None,
                 images=(), special=False):
        """Blocking; on_text(str) gets text deltas (False stops); special: control tokens as text.
        Returns (text, ids, stats, finish_reason)."""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        hold = max((len(s) for s in stop_strings), default=1) - 1
        state = {"text": "", "sent": 0, "ids": [], "stopped": None, "gone": False}
        # --progress N: every N s the tokens so far, the last window's tok/s and
        # tokens per round
        every = getattr(Handler.args, "progress", 0) or 0
        lg = {"t0": None, "last": None, "n": 0, "wn": 0, "wr": 0}

        def flush(final):
            text = state["text"]
            end = len(text) if final else max(state["sent"], len(text) - hold)
            if end > state["sent"]:
                if on_text(text[state["sent"]:end]) is False:
                    state["gone"] = True
                state["sent"] = end

        def cb(_user, toks, n):
            lg["n"] += n
            if every > 0:
                now = time.time()
                if lg["t0"] is None:
                    lg["t0"] = lg["last"] = now
                else:
                    lg["wn"] += n
                    lg["wr"] += 1
                    if now - lg["last"] >= every:
                        log("  .. %d tok %.1f t/s %.2f/rnd" % (lg["n"], lg["wn"] / (now - lg["last"]),
                                                                  lg["wn"] / lg["wr"]))
                        lg["last"], lg["wn"], lg["wr"] = now, 0, 0
            for i in range(n):
                t = toks[i]
                if want_ids:
                    state["ids"].append(t)
                state["text"] += dec.decode(self.piece(t, special))
                if stop_strings:
                    cut = min((state["text"].find(s) for s in stop_strings if s in state["text"]), default=-1)
                    if cut >= 0:
                        state["text"] = state["text"][:cut]
                        state["stopped"] = "stop"
                        flush(True)
                        return False
            flush(False)
            return not state["gone"]

        arr = (ctypes.c_int32 * len(prompt))(*prompt)
        eog = [] if ignore_eos else self.eog
        stops = (ctypes.c_int32 * max(len(eog), 1))(*eog)
        st = QzStats()
        with self.lock:
            ids = (ctypes.c_int * max(len(images), 1))(*[i[0] for i in images])
            rc = self.L.qz_generate_mm(self.ctx, arr, len(prompt), ids, len(images), max_new, stops, len(eog),
                                       EMIT_CB(cb), None, ctypes.byref(st),
                                       ctypes.byref(sampling) if sampling is not None else None)
        state["text"] += dec.decode(b"", final=True)
        flush(True)
        if rc != 0:
            finish = "error"
        elif state["stopped"]:
            finish = "stop"
        elif st.stop in (1, 3):
            finish = "stop"
        else:
            finish = "length"
        return state["text"], state["ids"], st, finish


# QZ_TRACE=<dir>: one JSON file per request plus flags for answers that look
# cut short (read with quartz/server/quartz_trace.py)
TRACE = os.environ.get("QZ_TRACE")
ENGINE_STOP = {0: "max_new/arena", 1: "eog", 2: "abort", 3: "client gone", -1: "error"}
ENGINE_ERROR = {"error": {"message": "the engine failed during generation (see the server log)",
                          "type": "server_error", "code": 500}}


KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def kfd_arches():
    """gfx name per GPU node in HIP order, or None when the KFD topology is not
    readable. HIP numbers the GPU nodes in node order, skipping the CPU node."""
    try:
        nodes = sorted(int(d) for d in os.listdir(KFD_NODES) if d.isdigit())
    except OSError:
        return None
    arches = []
    for nd in nodes:
        try:
            text = open(os.path.join(KFD_NODES, str(nd), "properties")).read()
        except OSError:
            continue
        props = dict(line.split()[:2] for line in text.splitlines() if len(line.split()) == 2)
        if int(props.get("simd_count", "0") or 0) == 0:
            continue                                   # CPU node
        ver = int(props.get("gfx_target_version", "0") or 0)
        arches.append("gfx%d%d%d" % (ver // 10000, ver // 100 % 100, ver % 100))
    return arches


def select_devices():
    """Keep only the gfx906 cards in the visible set, before HIP initializes.

    The runtime binds kernel images to the first visible device, so a card of
    another architecture in front of the MI50s fails every launch even when the
    work runs on an MI50. Done here because the variable is read once, at the
    first HIP call; the engine cannot change it from inside."""
    if os.environ.get("ROCR_VISIBLE_DEVICES") and not os.environ.get("HIP_VISIBLE_DEVICES"):
        return                                         # filtered below HIP: leave it alone
    arches = kfd_arches()
    if not arches:
        return                                         # no topology to read: the engine validates
    allowed = list(range(len(arches)))
    want = os.environ.get("HIP_VISIBLE_DEVICES")
    if want:
        try:
            listed = [int(v) for v in want.split(",")]
        except ValueError:
            return                                     # UUIDs and the like: leave it alone
        allowed = [i for i in allowed if i in listed]
    keep = [i for i in allowed if arches[i] == "gfx906"]
    if not keep:
        return                                         # nothing to keep: the engine reports why
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in keep)
    if len(keep) < len(arches):
        log("devices: %d of %d are gfx906, using HIP %s (%s)" % (
            len(keep), len(arches), ",".join(str(i) for i in keep), " ".join(arches)))


def log(msg):
    sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


_ESC = "\ue000"
_SPECIAL_RE = re.compile(r"(<\|[^<>|\s]+\|>)")


def escape_specials(x):
    """Escape "<|" in message / tool strings so quoted control tokens stay text."""
    if isinstance(x, str):
        return x.replace("<|", _ESC)
    if isinstance(x, list):
        return [escape_specials(v) for v in x]
    if isinstance(x, dict):
        return {k: escape_specials(v) for k, v in x.items()}
    return x


def normalize_messages(messages):
    """OpenAI chat messages -> what the GGUF template expects: content None ->
    "", assistant tool-call arguments as objects (clients send JSON strings)."""
    out = []
    for m in messages or []:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                f = dict(tc.get("function") or {})
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a) if a.strip() else {}
                    except ValueError:
                        a = {"arguments": a}
                    f["arguments"] = a if isinstance(a, dict) else {"arguments": a}
                tc["function"] = f
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def extract_images(messages):
    """The image parts' bytes, in prompt order (data: URLs only)."""
    blobs = []
    for m in messages or []:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict) or not (part.get("type") in ("image_url", "image") or "image_url" in part):
                continue
            u = part.get("image_url")
            u = u.get("url") if isinstance(u, dict) else (u or part.get("image"))
            if not isinstance(u, str) or not u.startswith("data:") or "," not in u:
                raise ValueError("images must be data: URLs (base64)")
            head, data = u.split(",", 1)
            blobs.append(base64.b64decode(data) if head.endswith(";base64") else urllib.parse.unquote_to_bytes(data))
    return blobs


def model_id(eng):
    """The /v1/models id: quartz for qwen35, else quartz-<general.name>."""
    if eng.arch == "qwen35":
        return "quartz"
    name = eng.info.name.decode(errors="replace") or eng.arch
    return "quartz-" + re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    eng = None
    args = None
    model_id = "quartz"

    def log_request(self, code="-", size="-"):     # only failed requests get an access line
        if isinstance(code, int) and code >= 400:
            log("http %s %s" % (code, self.requestline))

    def log_message(self, fmt, *a):
        log(fmt % a)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error", "code": code}})

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model_id, "object": "model",
                                                         "owned_by": "quartz", "max_ctx": self.eng.max_ctx}]})
        else:
            self._err(404, "not found")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._err(400, "bad json: %s" % e)
        if self.path == "/v1/chat/completions":
            return self._complete(req, chat=True)
        if self.path == "/v1/completions":
            return self._complete(req, chat=False)
        self._err(404, "not found")

    def _sampling(self, req):
        """(QzSampling or None for greedy, error string or None, notes)"""
        a = self.args
        def num(key, dflt, cast):
            v = req.get(key)
            return dflt if v is None else cast(v)
        try:
            temp = num("temperature", a.temp, float)
            top_k = num("top_k", a.top_k, int)
            top_p = num("top_p", a.top_p, float)
            min_p = num("min_p", a.min_p, float)
            seed = req.get("seed")
        except (TypeError, ValueError) as e:
            return None, "sampling params: %s" % e, []
        notes = []
        for key, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repeat_penalty", 1),
                             ("typical_p", 1), ("dry_multiplier", 0), ("xtc_probability", 0)):
            v = req.get(key)
            if v is not None and float(v) != neutral:
                notes.append("%s not applied" % key)
        for key in ("logit_bias", "grammar", "json_schema", "response_format"):
            if req.get(key):
                notes.append("%s not applied" % key)
        if temp <= 0 or top_k == 1:
            return None, None, notes
        if not 1 <= top_k <= 64:
            return None, "top_k must be 1..64 with temperature > 0 (the engine keeps the top 64 logits per row)", notes
        s = QzSampling()
        s.temp, s.top_k, s.top_p, s.min_p, s.min_keep = temp, top_k, top_p, min_p, 0
        s.seed = (int(seed) & 0xFFFFFFFF) if seed is not None and int(seed) >= 0 else random.getrandbits(32)
        s.n_order = 4
        for i, o in enumerate((2, 3, 4, 7)):    # llama.cpp's default order: top-k, top-p, min-p, temperature
            s.order[i] = o
        return s, None, notes

    # reasoning_effort (qwen35: xhigh / medium / low; other models: the request's value as
    # is): chat_template_kwargs, then the request's reasoning_effort, then --reasoning-effort
    _EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}

    def _template_kwargs(self, req):
        kw = dict(req.get("chat_template_kwargs") or {})
        if "reasoning_effort" not in kw:
            eff = self._EFFORT.get(str(req.get("reasoning_effort") or "").lower()) or self.args.reasoning_effort
            if self.eng.arch != "qwen35" and req.get("reasoning_effort"):
                eff = req["reasoning_effort"]      # other templates take their own levels (Inkling: none .. max)
            if eff:
                kw["reasoning_effort"] = eff
        return kw

    def _complete(self, req, chat):
        sampling, err, notes = self._sampling(req)
        if err:
            return self._err(400, err)
        tr = None
        try:
            think_open = False
            text = None
            imgs = []
            if chat:
                blobs = extract_images(req.get("messages"))
                if blobs and not self.eng.vision:
                    return self._err(400, "image input needs the server started with --mmproj")
                text = self.eng.render(escape_specials(normalize_messages(req.get("messages", []))),
                                       escape_specials(req.get("tools")),
                                       self._template_kwargs(req))
                think_open = self.eng.dialect.think_open(text)
                prompt = self.eng.tokenize_chat(text)
                text = text.replace(_ESC, "<|")
                if blobs:
                    imgs = self.eng.add_images(blobs)
                    prompt = self.eng.expand_images(prompt, imgs)
            else:
                p = req.get("prompt", "")
                if isinstance(p, list) and p and isinstance(p[0], str):
                    p = p[0]
                prompt = [int(x) for x in p] if isinstance(p, list) else self.eng.tokenize(p)
        except Exception as e:
            return self._err(400, "prompt: %s" % e)
        if not prompt:
            return self._err(400, "empty prompt")
        if TRACE:
            tr = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "request": req, "rendered": text,
                  "think_open": think_open, "n_prompt": len(prompt)}
        max_new = req.get("max_completion_tokens") or req.get("max_tokens") or (4096 if chat else 16)
        room = self.eng.max_ctx - len(prompt)
        if room < 1:
            return self._err(400, "prompt of %d tokens exceeds the engine context %d" % (len(prompt), self.eng.max_ctx))
        max_new = max(0, min(int(max_new), room))
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        # how much of this prompt matches the previous one (the prefix cache can
        # reuse at most that much)
        prev = getattr(Handler, "_last_prompt", [])
        common = 0
        for a, b in zip(prev, prompt):
            if a != b:
                break
            common += 1
        Handler._last_prompt = prompt
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        want_ids = bool(req.get("return_tokens"))
        if tr is not None:
            tr["rid"], tr["common"] = rid, common
        ign = bool(req.get("ignore_eos"))     # llama-server's option: decode past end-of-generation ids
        if req.get("stream"):
            return self._stream(rid, created, chat, prompt, max_new, stop, want_ids, think_open, ign, sampling,
                                req.get("tools") if chat else None, tr, imgs)
        text, ids, st, finish = self.eng.generate(prompt, max_new, lambda s: True, stop, want_ids or tr is not None, ign,
                                                  sampling, imgs, chat and self.eng.dialect.special_pieces)
        usage = {"prompt_tokens": len(prompt), "completion_tokens": st.n_gen,
                 "total_tokens": len(prompt) + st.n_gen}
        if chat:
            reasoning, content, calls = self.eng.dialect.parse(text, think_open, req.get("tools"))
            msg = {"role": "assistant", "content": content if (content or not calls) else None}
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
                finish = "tool_calls"
            choice = {"index": 0, "message": msg, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
        self._summary(st, choice["finish_reason"])
        self._trace(tr, chat, text, ids, st, choice["finish_reason"], think_open, req.get("tools"), max_new)
        if finish == "error":
            return self._json(500, ENGINE_ERROR)
        out = {"id": rid, "object": obj, "created": created, "model": self.model_id,
               "choices": [choice], "usage": usage, "timings": self._timings(st)}
        if want_ids:
            out["tokens"] = ids
        if notes:
            out["quartz_note"] = "; ".join(notes)
        self._json(200, out)

    def _trace(self, tr, chat, text, ids, st, finish, think_open, tools, max_new):
        """Write the QZ_TRACE record; flag answers that look cut short."""
        if tr is None:
            return
        try:
            reasoning, content, calls = None, text, []
            if chat:
                reasoning, content, calls = self.eng.dialect.parse(text, think_open, tools)
            flags = []
            if st.stop != 1:
                flags.append("engine stop: " + ENGINE_STOP.get(st.stop, str(st.stop)))
            if not calls:
                c = (content or "").rstrip()
                if not c:
                    flags.append("empty content, no tool call")
                elif c.endswith(":"):
                    flags.append("ends with ':' and no tool call")
            if chat:
                flags += self.eng.dialect.flags(text, think_open, content, calls)
            tr.update({"finish": finish, "engine_stop": st.stop, "max_new": max_new, "n_gen": st.n_gen,
                       "n_cached": st.n_cached, "raw": self.eng.detok(ids), "ids": ids,
                       "reasoning": reasoning, "content": content, "tool_calls": calls, "flags": flags})
            os.makedirs(TRACE, exist_ok=True)
            path = os.path.join(TRACE, "%s-%s.json" % (time.strftime("%Y%m%d-%H%M%S"), tr["rid"][-8:]))
            with open(path, "w") as f:
                json.dump(tr, f, ensure_ascii=False, indent=1)
            shown = [f for f in flags if not (f == "engine stop: max_new/arena" and finish == "length")]
            if shown:
                log("flag %s: %s" % (os.path.basename(path)[9:-5], "; ".join(shown)))
        except Exception as e:                  # tracing never breaks a request
            log("trace failed: %r" % e)

    def _summary(self, st, finish):
        t = self._timings(st)
        log("%d tok %.1f t/s %.2f/rnd %s | prompt %d, %d cached, %.1f s%s" % (
            t["predicted_n"], t["predicted_per_second"], t["tokens_per_cycle"], self.eng.drafter_name(st.dspine),
            st.n_prompt, t["cache_n"], t["prompt_ms"] / 1000, "" if finish == "stop" else " | " + finish))

    def _timings(self, st):
        # llama-server's convention: prompt_n = tokens processed, cache_n = taken from the prefix cache
        return {"cache_n": st.n_cached, "prompt_n": st.n_prompt - st.n_cached, "prompt_ms": round(st.prefill_ms, 1),
                "prompt_per_second": round(1000.0 * (st.n_prompt - st.n_cached) / st.prefill_ms, 1) if st.prefill_ms else 0,
                "predicted_n": st.n_gen, "predicted_ms": round(st.decode_ms, 1),
                "predicted_per_second": round(1000.0 * max(st.n_gen - 1, 0) / st.decode_ms, 1) if st.decode_ms else 0,
                "tokens_per_cycle": round(st.decode_tok / st.decode_cycles, 2) if st.decode_cycles else 0}

    def _stream(self, rid, created, chat, prompt, max_new, stop, want_ids, think_open=False, ign=False, sampling=None,
                tools=None, tr=None, imgs=()):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        obj = "chat.completion.chunk" if chat else "text_completion"

        def send(choice, extra=None):
            d = {"id": rid, "object": obj, "created": created, "model": self.model_id, "choices": [choice]}
            if extra:
                d.update(extra)
            try:
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        if chat:
            send({"index": 0, "delta": {"role": "assistant"}, "finish_reason": None})

        def delta(key):
            return lambda s: send({"index": 0, "delta": {key: s}, "finish_reason": None})

        def send_calls(calls):
            return send({"index": 0, "delta": {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]},
                         "finish_reason": None})

        if chat:
            ps = self.eng.dialect.stream(think_open, tools, delta("reasoning_content"), delta("content"), send_calls)
            on_text = ps.feed
        else:
            def on_text(s):
                return send({"index": 0, "text": s, "finish_reason": None})

        text, ids, st, finish = self.eng.generate(prompt, max_new, on_text, stop, want_ids or tr is not None, ign, sampling, imgs,
                                                  chat and self.eng.dialect.special_pieces)
        if chat and ps.close():
            finish = "tool_calls"
        self._summary(st, finish)
        self._trace(tr, chat, text, ids, st, finish, think_open, tools, max_new)
        if finish == "error":                    # headers are out: the error goes as an SSE event
            try:
                self.wfile.write(b"data: " + json.dumps(ENGINE_ERROR).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        last = {"index": 0, "finish_reason": finish}
        last["delta" if chat else "text"] = {} if chat else ""
        extra = {"usage": {"prompt_tokens": len(prompt), "completion_tokens": st.n_gen,
                           "total_tokens": len(prompt) + st.n_gen}, "timings": self._timings(st)}
        if want_ids:
            extra["tokens"] = ids
        send(last, extra)
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def preflight(args):
    """Fail fast, in words, on what would otherwise surface a minute into the load."""
    def die(msg):
        log("error: " + msg)
        sys.exit(2)
    if not os.path.isfile(args.lib):
        die("%s not found: run `make` first (or pass --lib)" % args.lib)
    for flag, path in (("--model", args.model), ("--dspine", args.dspine), ("--mmproj", args.mmproj)):
        if not path:
            continue
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
        except OSError as e:
            die("%s %s: %s" % (flag, path, e.strerror or e))
        if magic != b"GGUF":
            die("%s %s is not a GGUF file (is the download complete?)" % (flag, path))
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as the server binds: TIME_WAIT is not busy
    try:
        probe.bind((args.host, args.port))
    except OSError as e:
        die("cannot listen on %s:%d (%s): another server running? pass --port" % (args.host, args.port, e.strerror or e))
    finally:
        probe.close()


def supervise(args):
    """Run the server as a child and start it again when its start crashes or
    hangs (HIP initialisation occasionally does). Exceptions the child reports
    itself, such as a refused model, are not retried; QZ_START_RETRIES (default
    3) caps the attempts and QZ_START_TIMEOUT (default 180 s) the wait for ready."""
    tries = max(1, int(os.environ.get("QZ_START_RETRIES", "3")))
    wait = float(os.environ.get("QZ_START_TIMEOUT", "180"))
    stop = []

    def on_signal(n, _frame):
        stop.append(n)                    # asked to quit: no restart
        if child.poll() is None:
            child.send_signal(n)
    for attempt in range(1, tries + 1):
        rd, wr = os.pipe()
        env = dict(os.environ, QZ_CHILD_READY_FD=str(wr))
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env, pass_fds=(wr,))
        os.close(wr)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_signal)
        ready = timed_out = False
        try:
            if select.select([rd], [], [], wait)[0]:
                ready = os.read(rd, 1) == b"1"    # EOF instead: the child died
            else:
                timed_out = True
        except InterruptedError:
            pass
        os.close(rd)
        if ready:
            sys.exit(child.wait())
        if timed_out:
            log("start: no ready after %.0f s, killing it" % wait)
            child.kill()
        rc = child.wait()
        if stop or rc >= 0:               # asked to quit, or the child reported its own error: retrying will not help
            sys.exit(rc)
        if attempt < tries:
            log("start: the server died on signal %d before ready (HIP init fault?), retrying (%d/%d)" % (-rc, attempt + 1, tries))
        else:
            log("start: failed %d times, giving up" % tries)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="target GGUF (a supported architecture: docs/adding-a-model.md)")
    ap.add_argument("--lib", default=os.path.join(REPO, "build/bin/libquartz.so"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-ctx", type=int, default=262144, help="context length (262144: 29.7 GB per card)")
    ap.add_argument("--slot-t", type=int, default=0, help="rows per cycle: 3, 4, or 0 adaptive")
    ap.add_argument("--slack", type=int, default=1, help="adaptive depth slack")
    ap.add_argument("--temp", type=float, default=1.0, help="default temperature (0 = greedy)")
    ap.add_argument("--top-k", type=int, default=20, help="default top-k (1..64 when sampling)")
    ap.add_argument("--top-p", type=float, default=0.95, help="default top-p")
    ap.add_argument("--min-p", type=float, default=0.0, help="default min-p")
    ap.add_argument("--mmproj", default=None, help="vision projector GGUF (Qwen: qwen3vl_merger, Inkling: inkling2v): image input in chat")
    ap.add_argument("--dspine", default=None, help="DSpine draft GGUF: decode with the block drafter")
    ap.add_argument("--progress", type=float, default=0, metavar="SECONDS",
                    help="log decode progress every SECONDS (0: off)")
    ap.add_argument("--reasoning-effort", default="medium", choices=["xhigh", "medium", "low", "template"],
                    help="default chat-template reasoning_effort when the request names none (template: the template's own)")
    args = ap.parse_args()
    if args.max_ctx < 1024:
        ap.error("--max-ctx must be at least 1024")
    if args.reasoning_effort == "template":
        args.reasoning_effort = None
    if args.dspine:                          # read by qz_create
        os.environ["QZ_DSPINE"] = args.dspine
    if args.mmproj:                          # read by qz_create's memory check
        os.environ["QZ_MMPROJ"] = args.mmproj
    # read by the HIP runtime when libquartz first touches a device: P2P over PCIe
    # needs fine-grained PCIe memory
    os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "8")
    preflight(args)
    if "QZ_CHILD_READY_FD" not in os.environ and os.environ.get("QZ_START_RETRIES") != "0":
        supervise(args)
    select_devices()
    t0 = time.time()
    try:
        Handler.eng = Engine(args.lib, args.model, args.max_ctx, args.slot_t, args.slack)
    except RuntimeError as e:
        log("error: %s (the lines above say why; usual causes: the target has no MTP block, "
            "the cards are not two gfx906 with 32 GB free, or --max-ctx is too large)" % e)
        sys.exit(1)
    if args.mmproj:
        Handler.eng.attach_vision(args.mmproj)
    Handler.args = args
    Handler.model_id = model_id(Handler.eng)
    log("ready in %.1f s | ctx %d | %s%s | %s:%d%s" % (
        time.time() - t0, Handler.eng.max_ctx, Handler.eng.drafter_name(bool(args.dspine)), " + vision" if args.mmproj else "",
        args.host, args.port, " | trace " + TRACE if TRACE else ""))
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    fd = os.environ.get("QZ_CHILD_READY_FD")
    if fd:
        os.write(int(fd), b"1")           # the supervisor stops waiting
        os.close(int(fd))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
