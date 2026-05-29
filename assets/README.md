## Assets Directory

This directory contains vendored C headers from llama.cpp and their license files.

The headers are reference files. They show which version of the llama.cpp C API this Crystal binding was written against.

They are not used to build the shard. llama.cr links to the installed `libllama`.

These files are useful when checking version mismatches between the Crystal binding and the installed llama.cpp library.
