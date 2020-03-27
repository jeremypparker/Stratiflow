#pragma once
#include "Stratiflow.h"
#include <string>

std::string ExecuteShell(const std::string& cmd);

void MakeCleanDir(const std::string& dir);

void MoveDirectory(const std::string& from, const std::string& to);

void SaveValueToFile(stratifloat value, const std::string& filename);
void SaveValuesToFile(const std::vector<stratifloat>& values, const std::string& filename);
void LoadValueFromFile(stratifloat& value, const std::string& filename);

template<typename T>
void LoadVariable(const std::string& filename, T& into, int index)
{
    std::ifstream filestream(filename, std::ios::in | std::ios::binary);

    filestream.seekg(gridParams.N1*gridParams.N2*gridParams.N3*index*sizeof(stratifloat));
    into.Load(filestream);
}

bool FileExists(const std::string& filename);

inline bool EndsWith(const std::string& value, const std::string& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
