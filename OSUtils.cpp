#include "OSUtils.h"

#include <array>
#include <string>
#include <stdexcept>
#include <memory>
#include <fstream>

std::string ExecuteShell(const std::string& cmd)
{
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
    ExecuteShell("rm -rf " + dir);
    ExecuteShell("mkdir -p " + dir);
}

void MoveDirectory(const std::string& from, const std::string& to)
{
    ExecuteShell("rm -rf " + to);
    ExecuteShell("mv " + from + " " + to);
}
