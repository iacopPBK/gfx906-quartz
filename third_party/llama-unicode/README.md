llama.cpp's MIT-licensed unicode tables and pre-tokenizer helpers, vendored
unchanged from src/unicode.{h,cpp} and src/unicode-data.{h,cpp} so Quartz's
tokenizer needs no llama.cpp tree.

Source: https://github.com/ggml-org/llama.cpp (see LICENSE, copyright
"The ggml authors"). Quartz passes the pre-tokenizer regexes named by the
GGUF's tokenizer.ggml.pre (table in quartz/tokapi.cpp) to unicode_regex_split():
qwen35 dispatches to unicode.cpp's custom splitter, gpt-4o / inkling2 run on
the generic std::regex path.
