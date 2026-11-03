#ifndef FILES_HPP
#define FILES_HPP

#include <filesystem>
#include <system_error>

namespace othello_bot {

// Calls write(tmp) to write the file at a temporary path next to `path`, then
// renames it over `path` in one step, so an interrupted write leaves the
// previous file (if any) intact and readers never see a half-written one.
// The temporary file is removed if write throws.
template <class Write>
void write_atomically(const std::filesystem::path& path, Write write) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    try {
        write(tmp);
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

} // namespace othello_bot

#endif // FILES_HPP
