llama.cpp's MIT-licensed unicode tables and pre-tokenizer helpers, vendored
unchanged from src/unicode.{h,cpp} and src/unicode-data.{h,cpp} so Quartz's
tokenizer needs no llama.cpp tree.

Source: https://github.com/ggml-org/llama.cpp (see LICENSE, copyright
"The ggml authors"). Quartz passes the qwen35 pre-tokenizer regex to
unicode_regex_split(), which dispatches to unicode.cpp's custom qwen35
splitter.
