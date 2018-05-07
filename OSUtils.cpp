#include "OSUtils.h"

#include <array>
#include <string>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include <limits.h>
#include <errno.h>

int mkdir_p(const char *path)
{
    /* Adapted from http://stackoverflow.com/a/2336245/119527 */
    const size_t len = strlen(path);
    char _path[PATH_MAX];
    char *p;

    errno = 0;

    /* Copy string so its mutable */
    if (len > sizeof(_path)-1) {
        errno = ENAMETOOLONG;
        return -1;
    }
    strcpy(_path, path);

    /* Iterate the string */
    for (p = _path + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';

            if (mkdir(_path, S_IRWXU) != 0) {
                if (errno != EEXIST)
                    return -1;
            }

            *p = '/';
        }
    }

    if (mkdir(_path, S_IRWXU) != 0) {
        if (errno != EEXIST)
            return -1;
    }

    return 0;
}

std::string ExecuteShell(const std::string& cmd)
{
    std::cout << cmd << std::endl;
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

void MakeCleanDir(const std::string& dir)
{
    mkdir_p(dir.c_str());
}

void MoveDirectory(const std::string& from, const std::string& to)
{
    ExecuteShell("rm -rf " + to);
    ExecuteShell("mv " + from + " " + to);
}

bool FileExists(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    return file.good();
}
