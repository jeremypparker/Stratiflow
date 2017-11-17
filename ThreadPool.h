#pragma once

#include <thread>
#include <vector>
#include <functional>

constexpr int maxthreads = 3;

class ThreadPool
{
public:
    static ThreadPool& Get()
    {
        static ThreadPool instance;
        return instance;
    }

public:
    void ExecuteAsync(std::function<void()> f)
    {
        //f();
        if(threads[next].joinable())
        {
            threads[next].join();
        }

        threads[next] = std::thread(f);

        next++;
        if (next == numthreads)
        {
            next = 0;
        }

    }

    void WaitAll()
    {
        for (int i=0; i<numthreads; i++)
        {
            if(threads[i].joinable())
            {
                threads[i].join();
            }
        }
    }
private:
    ThreadPool(int nthreads=maxthreads)
    : numthreads(nthreads)
    , threads(numthreads)
    {}

    ~ThreadPool()
    {
        WaitAll();
    }

    int numthreads;
    std::vector<std::thread> threads;
    int next = 0;
};