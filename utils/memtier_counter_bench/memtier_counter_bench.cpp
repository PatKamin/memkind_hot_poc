// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2021 Intel Corporation. */

#include <memkind/internal/memkind_memtier.h>

#include <argp.h>
#include <assert.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <stdint.h>
#include <thread>
#include <vector>

class counter_bench_alloc;

using Benchptr = std::unique_ptr<counter_bench_alloc>;
struct BenchArgs {
    Benchptr bench;
    size_t thread_no;
    size_t run_no;
    size_t iter_no;
};

class counter_bench_alloc
{
public:
    double run(BenchArgs& arguments) const
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

        if (arguments.thread_no == 1) {
            for (size_t r = 0; r < arguments.run_no; ++r) {
                single_run(arguments);
            }
        } else {
            std::vector<std::thread> vthread(arguments.thread_no);
            for (size_t r = 0; r < arguments.run_no; ++r) {
                for (size_t k = 0; k < arguments.thread_no; ++k) {
                    vthread[k] = std::thread([&]() { single_run(arguments); });
                }
                for (auto &t : vthread) {
                    t.join();
                }
            }
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end-start;
        auto millis_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                .count();

        auto time_per_op =
            ((double)millis_elapsed) / arguments.iter_no;
        return time_per_op / (arguments.run_no * arguments.thread_no);
    }
    virtual ~counter_bench_alloc() = default;

protected:
    const size_t m_size = 512;
    virtual void *bench_alloc(size_t) const = 0;
    virtual void bench_free(void *) const = 0;

private:
    void single_run(BenchArgs& arguments) const
    {
        std::vector<void *> v;
        v.reserve(arguments.iter_no);
        for (size_t i = 0; i < arguments.iter_no; i++) {
            v.emplace_back(bench_alloc(m_size));
        }
        for (size_t i = 0; i < arguments.iter_no; i++) {
            bench_free(v[i]);
        }
        v.clear();
    }
};

class memkind_bench_alloc: public counter_bench_alloc
{
protected:
    void *bench_alloc(size_t size) const final
    {
        return memkind_malloc(MEMKIND_DEFAULT, size);
    }

    void bench_free(void *ptr) const final
    {
        memkind_free(MEMKIND_DEFAULT, ptr);
    }
};

class memtier_kind_bench_alloc: public counter_bench_alloc
{
protected:
    void *bench_alloc(size_t size) const final
    {
        return memtier_kind_malloc(MEMKIND_DEFAULT, size);
    }

    void bench_free(void *ptr) const final
    {
        memtier_kind_free(MEMKIND_DEFAULT, ptr);
    }
};

class memtier_bench_alloc: public counter_bench_alloc
{
public:
    memtier_bench_alloc()
    {
        m_tier_builder = memtier_builder_new(MEMTIER_POLICY_STATIC_RATIO);
        memtier_builder_add_tier(m_tier_builder, MEMKIND_DEFAULT, 1);
        m_tier_memory =
            memtier_builder_construct_memtier_memory(m_tier_builder);
    }

    ~memtier_bench_alloc()
    {
        memtier_builder_delete(m_tier_builder);
        memtier_delete_memtier_memory(m_tier_memory);
    }

protected:
    void *bench_alloc(size_t size) const final
    {
        return memtier_malloc(m_tier_memory, size);
    }

    void bench_free(void *ptr) const final
    {
        memtier_realloc(m_tier_memory, ptr, 0);
    }

private:
    struct memtier_builder *m_tier_builder;
    struct memtier_memory *m_tier_memory;
};

class memtier_multiple_bench_alloc: public counter_bench_alloc
{
public:
    memtier_multiple_bench_alloc(memtier_policy_t policy)
    {
        m_tier_builder = memtier_builder_new(policy);
        memtier_builder_add_tier(m_tier_builder, MEMKIND_DEFAULT, 1);
        memtier_builder_add_tier(m_tier_builder, MEMKIND_REGULAR, 1);
        m_tier_memory =
            memtier_builder_construct_memtier_memory(m_tier_builder);
    }

    ~memtier_multiple_bench_alloc()
    {
        memtier_builder_delete(m_tier_builder);
        memtier_delete_memtier_memory(m_tier_memory);
    }

protected:
    void *bench_alloc(size_t size) const final
    {
        return memtier_malloc(m_tier_memory, size);
    }

    void bench_free(void *ptr) const final
    {
        memtier_realloc(m_tier_memory, ptr, 0);
    }

private:
    struct memtier_builder *m_tier_builder;
    struct memtier_memory *m_tier_memory;
};

// clang-format off
static int parse_opt(int key, char *arg, struct argp_state *state)
{
    auto args = (BenchArgs *)state->input;
    switch (key) {
        case 'm':
            args->bench = Benchptr(new memkind_bench_alloc());
            break;
        case 'k':
            args->bench = Benchptr(new memtier_kind_bench_alloc());
            break;
        case 'x':
            args->bench = Benchptr(new memtier_bench_alloc());
            break;
        case 's':
            args->bench = Benchptr(new memtier_multiple_bench_alloc(MEMTIER_POLICY_STATIC_RATIO));
            break;
        case 'd':
            args->bench = Benchptr(new memtier_multiple_bench_alloc(MEMTIER_POLICY_DYNAMIC_THRESHOLD));
            break;
        case 'p':
            args->bench = Benchptr(new memtier_multiple_bench_alloc(MEMTIER_POLICY_DATA_HOTNESS));
            break;
        case 't':
            args->thread_no = std::strtol(arg, nullptr, 10);
            break;
        case 'r':
            args->run_no = std::strtol(arg, nullptr, 10);
        case 'i':
            args->iter_no = std::strtol(arg, nullptr, 10);
            break;
    }
    return 0;
}

static struct argp_option options[] = {
    {"memkind", 'm', 0, 0, "Benchmark memkind."},
    {"memtier_kind", 'k', 0, 0, "Benchmark memtier_memkind."},
    {"memtier", 'x', 0, 0, "Benchmark memtier_memory - single tier."},
    {"memtier_multiple", 's', 0, 0, "Benchmark memtier_memory - two tiers, static ratio."},
    {"memtier_multiple", 'd', 0, 0, "Benchmark memtier_memory - two tiers, dynamic threshold."},
    {"memtier_multiple", 'p', 0, 0, "Benchmark memtier_memory - two tiers, data hotness."},
    {"thread", 't', "int", 0, "Threads numbers."},
    {"runs", 'r', "int", 0, "Benchmark run numbers."},
    {"iterations", 'i', "int", 0, "Benchmark iteration numbers."},
    {0}};
// clang-format on

static struct argp argp = {options, parse_opt, nullptr, nullptr};

int main(int argc, char *argv[])
{
    struct BenchArgs arguments = {
        .bench = nullptr,
        .thread_no = 0,
        .run_no = 1,
        .iter_no = 10000000 };

    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    double time_per_op =
        arguments.bench->run(arguments);
    std::cout << "Mean milliseconds per operation:" << time_per_op << std::endl;
    return 0;
}
