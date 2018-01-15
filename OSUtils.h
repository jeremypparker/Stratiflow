#pragma once
#include "Stratiflow.h"
#include <string>

std::string ExecuteShell(const std::string& cmd);

void MakeCleanDir(const std::string& dir);

void LoadVariable(const std::string& filename, NField& into, int index);