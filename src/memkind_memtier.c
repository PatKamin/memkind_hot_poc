// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2021 Intel Corporation. */

#include <memkind/internal/memkind_arena.h>
#include <memkind/internal/memkind_log.h>
#include <memkind/internal/bthash.h>
#include <memkind/internal/memkind_memtier.h>
#include <memkind/internal/pebs.h>
#include <memkind/internal/tachanka.h>

#include "config.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <threads.h>
#include <execinfo.h>

#ifdef HAVE_STDATOMIC_H
#include <stdatomic.h>
#define MEMKIND_ATOMIC _Atomic
#else
#define MEMKIND_ATOMIC
#endif

#if PRINT_POLICY_LOG_STATISTICS_INFO
    static atomic_size_t g_successful_adds=0;
    static atomic_size_t g_failed_adds=0;
    static atomic_size_t g_successful_adds_malloc=0;
    static atomic_size_t g_failed_adds_malloc=0;
    static atomic_size_t g_successful_adds_realloc0=0;
    static atomic_size_t g_failed_adds_realloc0=0;
    static atomic_size_t g_successful_adds_realloc1=0;
    static atomic_size_t g_failed_adds_realloc1=0;
    static atomic_size_t g_successful_adds_free=0;
    static atomic_size_t g_failed_adds_free=0;
#endif

// clang-format off
#if defined(MEMKIND_ATOMIC_C11_SUPPORT)
#define memkind_atomic_increment(counter, val)                                 \
    atomic_fetch_add_explicit(&counter, val, memory_order_relaxed)
#define memkind_atomic_decrement(counter, val)                                 \
    atomic_fetch_sub_explicit(&counter, val, memory_order_relaxed)
#define memkind_atomic_set(counter, val)                                       \
    atomic_store_explicit(&counter, val, memory_order_relaxed)
#define memkind_atomic_get(src, dest)                                          \
    do {                                                                       \
        dest = atomic_load_explicit(&src, memory_order_relaxed);               \
    } while (0)
#define memkind_atomic_get_and_zeroing(src)                                    \
    atomic_exchange_explicit(&src, 0, memory_order_relaxed)
#elif defined(MEMKIND_ATOMIC_BUILTINS_SUPPORT)
#define memkind_atomic_increment(counter, val)                                 \
    __atomic_fetch_add(&counter, val, __ATOMIC_RELAXED)
#define memkind_atomic_decrement(counter, val)                                 \
    __atomic_fetch_sub(&counter, val, __ATOMIC_RELAXED)
#define memkind_atomic_set(counter, val)                                       \
    __atomic_store_n(&counter, val, __ATOMIC_RELAXED)
#define memkind_atomic_get(src, dest)                                          \
    do {                                                                       \
        dest = __atomic_load_n(&src, __ATOMIC_RELAXED);                        \
    } while (0)
#define memkind_atomic_get_and_zeroing(src)                                    \
    __atomic_exchange_n(&src, 0, __ATOMIC_RELAXED)
#elif defined(MEMKIND_ATOMIC_SYNC_SUPPORT)
#define memkind_atomic_increment(counter, val)                                 \
    __sync_fetch_and_add(&counter, val)
#define memkind_atomic_decrement(counter, val)                                 \
    __sync_fetch_and_sub(&counter, val)
#define memkind_atomic_get(src, dest)                                          \
    do {                                                                       \
        dest = __sync_sub_and_fetch(&src, 0);                                  \
    } while (0)
#define memkind_atomic_get_and_zeroing(src)                                    \
    do {                                                                       \
        dest = __sync_fetch_and_sub(&src, &src);                               \
    } while (0)
#define memkind_atomic_set(counter, val)                                       \
    __atomic_store_n(&counter, val, __ATOMIC_RELAXED)
#else
#error "Missing atomic implementation."
#endif

// Default values for DYNAMIC_THRESHOLD configuration
// TRIGGER       - threshold between tiers will be updated if a difference
//                 between current and desired ratio between these tiers is
//                 greater than TRIGGER value (in percents)
// DEGREE        - if an update is triggered, DEGREE is the value (in percents)
//                 by which threshold will change
// CHECK_CNT     - number of memory management operations that has to be made
//                 between ratio checks
// STEP          - default step (in bytes) between thresholds
#define THRESHOLD_TRIGGER   0.02 // 2%
#define THRESHOLD_DEGREE    0.15 // 15%
#define THRESHOLD_CHECK_CNT 20
#define THRESHOLD_STEP      1024

// Macro to get number of thresholds from parent object
#define THRESHOLD_NUM(obj) ((obj->cfg_size) - 1)

struct memtier_tier_cfg {
    memkind_t kind;   // Memory kind
    float kind_ratio; // Memory kind ratio
};

// Thresholds configuration - valid only for DYNAMIC_THRESHOLD policy
struct memtier_threshold_cfg {
    size_t val;                // Actual threshold level
    size_t min;                // Minimum threshold level
    size_t max;                // Maximum threshold level
    float exp_norm_ratio;      // Expected normalized ratio between two adjacent
                               // tiers
    float current_ratio_diff;  // Difference between actual and expected
                               // normalized ratio
};

struct memtier_builder {
    unsigned cfg_size;            // Number of Memory Tier configurations
    struct memtier_tier_cfg *cfg; // Memory Tier configurations
    struct memtier_threshold_cfg *thres; // Thresholds configuration for
                                         // DYNAMIC_THRESHOLD policy
    unsigned check_cnt; // Number of memory management operations that has to be
                        // made between ratio checks
    float trigger;      // Difference between ratios to update threshold
    float degree;       // % of threshold change in case of update
    // builder operations
    struct memtier_memory *(*create_mem)(struct memtier_builder *builder);
    int (*update_builder)(struct memtier_builder *builder);
    int (*ctl_set)(struct memtier_builder *builder, const char *name, const void *val);
};

struct memtier_memory {
    unsigned cfg_size;                   // Number of memory kinds
    struct memtier_tier_cfg *cfg;        // Memory Tier configuration
    struct memtier_threshold_cfg *thres; // Thresholds configuration for
                                         // DYNAMIC_THRESHOLD policy
    unsigned thres_check_cnt;            // Counter for doing thresholds check
    unsigned thres_init_check_cnt;       // Initial value of check_cnt
    float thres_trigger;                 // Difference between ratios to update
                                         // threshold
    float thres_degree; // % of threshold change in case of update
    int hot_tier_id;                     // ID of "hot" tier

    // memtier_memory operations
    memkind_t (*get_kind)(struct memtier_memory *memory, size_t size, uint64_t *data);
    void (*post_alloc)(uint64_t data, void *addr, size_t size);
    void (*update_cfg)(struct memtier_memory *memory);
};
// clang-format on

#define THREAD_BUCKETS  (256U)
#define FLUSH_THRESHOLD (51200)

static MEMKIND_ATOMIC long long t_alloc_size[MEMKIND_MAX_KIND][THREAD_BUCKETS];
static MEMKIND_ATOMIC size_t g_alloc_size[MEMKIND_MAX_KIND];

/* Declare weak symbols for allocator decorators */
extern void memtier_kind_malloc_post(struct memkind *, size_t, void **)
    __attribute__((weak));
extern void memtier_kind_calloc_post(struct memkind *, size_t, size_t, void **)
    __attribute__((weak));
extern void memtier_kind_posix_memalign_post(struct memkind *, void **, size_t,
                                             size_t, int *)
    __attribute__((weak));
extern void memtier_kind_realloc_post(struct memkind *, void *, size_t, void **)
    __attribute__((weak));
extern void memtier_kind_free_pre(void **) __attribute__((weak));
extern void memtier_kind_usable_size_post(void **, size_t)
    __attribute__((weak));

void memtier_reset_size(unsigned kind_id)
{
    unsigned bucket_id;
    for (bucket_id = 0; bucket_id < THREAD_BUCKETS; ++bucket_id) {
        memkind_atomic_set(t_alloc_size[kind_id][bucket_id], 0);
    }
    memkind_atomic_set(g_alloc_size[kind_id], 0);
}

// SplitMix64 hash
static inline unsigned t_hash_64(void)
{
    uint64_t x = (uint64_t)pthread_self();
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return (x ^ (x >> 31)) & (THREAD_BUCKETS - 1);
}

static inline void increment_alloc_size(unsigned kind_id, size_t size)
{
    unsigned bucket_id = t_hash_64();
    if ((memkind_atomic_increment(t_alloc_size[kind_id][bucket_id], size) +
         size) > FLUSH_THRESHOLD) {
        size_t size_f =
            memkind_atomic_get_and_zeroing(t_alloc_size[kind_id][bucket_id]);
        memkind_atomic_increment(g_alloc_size[kind_id], size_f);
    }
}

static inline void decrement_alloc_size(unsigned kind_id, size_t size)
{
    unsigned bucket_id = t_hash_64();
    if ((memkind_atomic_decrement(t_alloc_size[kind_id][bucket_id], size) -
         size) < -FLUSH_THRESHOLD) {
        long long size_f =
            memkind_atomic_get_and_zeroing(t_alloc_size[kind_id][bucket_id]);
        memkind_atomic_increment(g_alloc_size[kind_id], size_f);
    }
}

static memkind_t memtier_single_get_kind(struct memtier_memory *memory,
                                         size_t size, uint64_t *data)
{
    return memory->cfg[0].kind;
}

static memkind_t
memtier_policy_static_ratio_get_kind(struct memtier_memory *memory,
                                     size_t size, uint64_t *data)
{
    struct memtier_tier_cfg *cfg = memory->cfg;

    size_t size_tier, size_0;
    int i;
    int dest_tier = 0;
    memkind_atomic_get(g_alloc_size[cfg[0].kind->partition], size_0);
    for (i = 1; i < memory->cfg_size; ++i) {
        memkind_atomic_get(g_alloc_size[cfg[i].kind->partition], size_tier);
        if ((size_tier * cfg[i].kind_ratio) < size_0) {
            dest_tier = i;
        }
    }
    return cfg[dest_tier].kind;
}

static memkind_t
memtier_policy_dynamic_threshold_get_kind(struct memtier_memory *memory,
                                          size_t size, uint64_t* data)
{
    struct memtier_threshold_cfg *thres = memory->thres;
    int i;

    for (i = 0; i < THRESHOLD_NUM(memory); ++i) {
        if (size < thres[i].val) {
            break;
        }
    }
    return memory->cfg[i].kind;
}

static bool memtier_policy_data_hotness_is_hot(uint64_t hash)
{
    // TODO this requires more data
    // Currently, "ranking" and "tachanka" are de-facto singletons
    // can we deal with it?
    Hotness_e hotness = tachanka_get_hotness_type_hash(hash);
    int ret = 0;

    // DEBUG
#if PRINT_POLICY_LOG_STATISTICS_INFO
    static atomic_uint_fast16_t counter=0;
    static atomic_uint_fast64_t hotness_counter[3]= { 0 };
    const uint64_t interval=100000;
    if (++counter > interval) {
        struct timespec t;
        int ret = clock_gettime(CLOCK_MONOTONIC, &t);
        if (ret != 0) {
            log_fatal("critnib: ASSERT COUNTER FAILURE!\n");
            exit(-1);
        }

        double hotness_thresh = tachanka_get_hot_thresh();
        log_info("critnib: hotness thresh: %f, counters [hot, cold, unknown]: %lu %lu %lu, "
            "[seconds, nanoseconds]: [%ld, %ld]\nsuccess/fail: %lu, %lu",
            hotness_thresh, hotness_counter[0], hotness_counter[1], hotness_counter[2],
            t.tv_sec, t.tv_nsec, g_successful_adds, g_failed_adds);

        log_info("critnib: success/fail: malloc [%lu/%lu], realloc0 [%lu/%lu], "
            "realloc1 [%lu/%lu], free [%lu/%lu]",
            g_successful_adds_malloc, g_failed_adds_malloc,
            g_successful_adds_realloc0, g_failed_adds_realloc0,
            g_successful_adds_realloc1, g_failed_adds_realloc1,
            g_successful_adds_free, g_failed_adds_free);
        counter=0u;

#if PRINT_POLICY_BACKTRACE_INFO
        static thread_local bool in_progress=false;
        if (!in_progress) {
            in_progress = true;
            // backtrace
            const size_t BUFF_SIZE=30;
            void *buff[BUFF_SIZE];
            ret = backtrace(buff, BUFF_SIZE);
            char **strings = backtrace_symbols(buff, ret);
            for (int i=0; i<ret; ++i) {
                log_info("backtrace: %s", strings[i]);
            }
            free(strings);
            in_progress = false;
        }
#endif //PRINT_POLICY_BACKTRACE_INFO
    }
#endif // PRINT_POLICY_LOG_STATISTICS_INFO

#if PRINT_POLICY_LOG_STATISTICS_INFO
    ++hotness_counter[hotness];
#endif

    switch (hotness) {
        case HOTNESS_COLD:
            ret = 0;
            break;
        case HOTNESS_NOT_FOUND:
        case HOTNESS_HOT:
            ret = 1;
            break;
        default:
            log_fatal("critnib: invalid hotness enum val");
            exit(-1);
    }
    return ret;
}

static thread_local void *stack_bottom=NULL;
static thread_local bool stack_bottom_initialized=false;

// TODO check if calling pthread functions from the context of pthread_once is ok...
void initialize_stack_bottom(void) {
    size_t stack_size = 0; // value irrelevant
    pthread_attr_t attr;

    if (!stack_bottom_initialized) {
        stack_bottom_initialized = true;

        int ret = pthread_getattr_np(pthread_self(), &attr);
        if (ret) {
            log_fatal("pthread get stack failed!");
            exit(-1);
        }

        pthread_attr_getstack(&attr, &stack_bottom, &stack_size);
        pthread_attr_destroy(&attr);
    }
}

static memkind_t
memtier_policy_data_hotness_get_kind(struct memtier_memory *memory, size_t size,
                                     uint64_t *data)
{
    // -- recursion prevention
    void *foo=NULL; // value is irrelevant
    // corner case, which is not handled: actual stack is different from the one returned by pthread
    void *stack_top= &foo;
    initialize_stack_bottom();
//     int ret = pthread_once(&stack_bottom_init, initialize_stack_bottom);
//     assert(ret == 0);
    bthash_set_stack_range(stack_top, stack_bottom);
    *data = bthash(size);
    // TODO support for multiple tiers could be added
    // instead of bool (,mis hot), an index of memory tier could be returned
    int dest_tier = memtier_policy_data_hotness_is_hot(*data) ?
        memory->hot_tier_id : 1 - memory->hot_tier_id;

    //char buf[128];
    //if (write(1, buf, sprintf(buf, "hash %016zx size %zd is %s\n", *data, size,
    //               memtier_policy_data_hotness_is_hot(*data) ? "♨": "❄")));

    return memory->cfg[dest_tier].kind;
}

static void
memtier_policy_data_hotness_post_alloc(uint64_t hash, void *addr, size_t size)
{
//     static atomic_uint_fast16_t counter=0;
//     const uint64_t interval=1000;
//     if (++counter > interval) {
//         struct timespec t;
//         int ret = clock_gettime(CLOCK_MONOTONIC, &t);
//         if (ret != 0) {
//             printf("ASSERT TOUCH COUNTER FAILURE!\n");
//         }
//         assert(ret == 0);
//         printf("touch counter %lu hit, [seconds, nanoseconds]: [%ld, %ld]\n",
//             interval, t.tv_sec, t.tv_nsec);
//         counter=0u;
//     }
    // TODO: there are 2 lookups in hash_to_block - one from "get_kind" and
    // second here - this could be easily optimized
//     register_block(hash, addr, size);
//     touch(addr, 0, 1 /*called from malloc*/);

    EventEntry_t entry = {
        .type = EVENT_CREATE_ADD,
        .data.createAddData = {
            .hash = hash,
            .address = addr,
            .size = size,
        },
    };

    // copy is performed, passing pointer to stack is ok
    bool success = tachanka_ranking_event_push(&entry);

#if PRINT_POLICY_LOG_STATISTICS_INFO
    if (success) {
        g_successful_adds++;
        g_successful_adds_malloc++;
    } else {
        g_failed_adds++;
        g_failed_adds_malloc++;
    }
#else
    (void)success;
#endif
    // TODO assure that failure can be ignored/handle failure!
}

static void print_memtier_memory(struct memtier_memory *memory)
{
    int i;
    if (!memory) {
        log_info("Empty memtier memory");
        return;
    }
    log_info("Number of memory tiers %u", memory->cfg_size);
    for (i = 0; i < memory->cfg_size; ++i) {
        log_info("Tier %d - memory kind %s", i, memory->cfg[i].kind->name);
        log_info("Tier normalized ratio %f", memory->cfg[i].kind_ratio);
        log_info("Tier allocated size %zu",
                 memtier_kind_allocated_size(memory->cfg[i].kind));
    }
    if (memory->thres) {
        for (i = 0; i < THRESHOLD_NUM(memory); ++i) {
            log_info("Threshold %d - minimum %zu", i, memory->thres[i].min);
            log_info("Threshold %d - current value %zu", i,
                     memory->thres[i].val);
            log_info("Threshold %d - maximum %zu", i, memory->thres[i].max);
        }
    } else {
        log_info("No thresholds configuration found");
    }
    log_info("Threshold trigger value %f", memory->thres_trigger);
    log_info("Threshold degree value %f", memory->thres_degree);
    log_info("Threshold counter setting value %u",
             memory->thres_init_check_cnt);
    log_info("Threshold counter current value %u", memory->thres_check_cnt);
}

static void print_builder(struct memtier_builder *builder)
{
    int i;
    if (!builder) {
        log_info("Empty builder");
        return;
    }
    log_info("Number of memory tiers %u", builder->cfg_size);
    for (i = 0; i < builder->cfg_size; ++i) {
        log_info("Tier %d - memory kind %s", i, builder->cfg[i].kind->name);
        log_info("Tier normalized ratio %f", builder->cfg[i].kind_ratio);
    }
    if (builder->thres) {
        for (i = 0; i < THRESHOLD_NUM(builder); ++i) {
            log_info("Threshold %d - minimum %zu", i, builder->thres[i].min);
            log_info("Threshold %d - current value %zu", i,
                     builder->thres[i].val);
            log_info("Threshold %d - maximum %zu", i, builder->thres[i].max);
        }
    } else {
        log_info("No thresholds configuration found");
    }
    log_info("Threshold trigger value %f", builder->trigger);
    log_info("Threshold degree value %f", builder->degree);
    log_info("Threshold counter setting value %u", builder->check_cnt);
}

static void
memtier_policy_static_ratio_update_config(struct memtier_memory *memory)
{}

static void
memtier_policy_data_hotness_update_config(struct memtier_memory *memory)
{}

static void
memtier_empty_post_alloc(uint64_t data, void *addr, size_t size)
{}

static void
memtier_policy_dynamic_threshold_update_config(struct memtier_memory *memory)
{
    struct memtier_tier_cfg *cfg = memory->cfg;
    struct memtier_threshold_cfg *thres = memory->thres;
    int i;
    size_t prev_alloc_size, next_alloc_size;

    // do the ratio checks only every each thres_check_cnt
    if (--memory->thres_check_cnt > 0) {
        return;
    }

    // for every pair of adjacent tiers, check if distance between actual vs
    // desired ratio between them is above TRIGGER level and if so, change
    // threshold by CHANGE val
    // TODO optimize the loop to avoid redundant atomic_get in 3 or more tier
    // scenario
    for (i = 0; i < THRESHOLD_NUM(memory); ++i) {
        memkind_atomic_get(g_alloc_size[cfg[i].kind->partition],
                           prev_alloc_size);
        memkind_atomic_get(g_alloc_size[cfg[i + 1].kind->partition],
                           next_alloc_size);

        float current_ratio = -1;
        if (prev_alloc_size > 0) {
            current_ratio = (float)next_alloc_size / prev_alloc_size;
            float prev_ratio_diff = thres[i].current_ratio_diff;
            thres[i].current_ratio_diff =
                fabs(current_ratio - thres[i].exp_norm_ratio);
            if ((thres[i].current_ratio_diff < memory->thres_trigger) ||
                (thres[i].current_ratio_diff < prev_ratio_diff)) {
                // threshold needn't to be changed
                continue;
            }
        }

        // increase/decrease threshold value by thres_degree and clamp it to
        // (min, max) range
        size_t threshold = (size_t)ceilf(thres[i].val * memory->thres_degree);
        if ((prev_alloc_size == 0) ||
            (current_ratio > thres[i].exp_norm_ratio)) {
            size_t higher_threshold = thres[i].val + threshold;
            if (higher_threshold <= thres[i].max) {
                thres[i].val = higher_threshold;
            }
        } else {
            size_t lower_threshold = thres[i].val - threshold;
            if (lower_threshold >= thres[i].min) {
                thres[i].val = lower_threshold;
            }
        }
    }

    // reset threshold check counter
    memory->thres_check_cnt = memory->thres_init_check_cnt;
}

static inline struct memtier_memory *
memtier_memory_init(size_t tier_size, bool is_dynamic_threshold,
                    bool is_data_hotness)
{
    if (tier_size == 0) {
        log_err("No tier in builder.");
        return NULL;
    }

    struct memtier_memory *memory = jemk_malloc(sizeof(struct memtier_memory));
    if (!memory) {
        log_err("malloc() failed.");
        return NULL;
    }

    memory->cfg = jemk_calloc(tier_size, sizeof(struct memtier_tier_cfg));
    if (!memory->cfg) {
        log_err("calloc() failed.");
        jemk_free(memory);
        return NULL;
    }
    if (is_dynamic_threshold) {
        memory->get_kind = memtier_policy_dynamic_threshold_get_kind;
        memory->post_alloc = memtier_empty_post_alloc;
        memory->update_cfg = memtier_policy_dynamic_threshold_update_config;
        memory->thres_check_cnt = THRESHOLD_CHECK_CNT;
    } else if (is_data_hotness) {
        memory->get_kind = memtier_policy_data_hotness_get_kind;
        memory->post_alloc = memtier_policy_data_hotness_post_alloc;
        memory->update_cfg = memtier_policy_data_hotness_update_config;
    } else {
        if (tier_size == 1)
            memory->get_kind = memtier_single_get_kind;
        else {
            memory->get_kind = memtier_policy_static_ratio_get_kind;
        }
        memory->post_alloc = memtier_empty_post_alloc;
        memory->update_cfg = memtier_policy_static_ratio_update_config;
    }
    memory->thres = NULL;
    memory->cfg_size = tier_size;

    return memory;
}

static struct memtier_memory *
builder_static_create_memory(struct memtier_builder *builder)
{
    int i;
    struct memtier_memory *memory =
        memtier_memory_init(builder->cfg_size, false, false);

    if (!memory) {
        log_err("memtier_memory_init failed.");
        return NULL;
    }

    for (i = 1; i < memory->cfg_size; ++i) {
        memory->cfg[i].kind = builder->cfg[i].kind;
        memory->cfg[i].kind_ratio =
            builder->cfg[0].kind_ratio / builder->cfg[i].kind_ratio;
    }
    memory->cfg[0].kind = builder->cfg[0].kind;
    memory->cfg[0].kind_ratio = 1.0;

    return memory;
}

static int builder_static_ctl_set(struct memtier_builder *builder,
                                  const char *name, const void *val)
{
    log_err("Invalid name: %s", name);
    return -1;
}

static int builder_hot_ctl_set(struct memtier_builder *builder,
                               const char *name, const void *val)
{
    log_err("Invalid name: %s", name);
    return -1;
}

static int builder_dynamic_ctl_set(struct memtier_builder *builder,
                                   const char *name, const void *val)
{
    const char *query = name;
    char name_substr[256] = {0};
    int chr_read = 0;

    sscanf(query, "policy.dynamic_threshold.%n", &chr_read);
    if (chr_read == sizeof("policy.dynamic_threshold.") - 1) {
        query += chr_read;

        int ret = sscanf(query, "%[^\[]%n", name_substr, &chr_read);
        if (ret && strcmp(name_substr, "thresholds") == 0) {
            query += chr_read;

            int th_indx = -1;
            ret = sscanf(query, "[%d]%n", &th_indx, &chr_read);
            if (th_indx >= 0) {
                if (th_indx + 1 >= builder->cfg_size) {
                    log_err("Too small tiers defined %d, for tier index %d",
                            builder->cfg_size, th_indx);
                    return -1;
                }
                query += chr_read;
                struct memtier_threshold_cfg *thres = &builder->thres[th_indx];

                ret = sscanf(query, ".%s", name_substr);
                if (ret && strcmp(name_substr, "val") == 0) {
                    thres->val = *(size_t *)val;
                    return 0;
                } else if (ret && strcmp(name_substr, "min") == 0) {
                    thres->min = *(size_t *)val;
                    return 0;
                } else if (ret && strcmp(name_substr, "max") == 0) {
                    thres->max = *(size_t *)val;
                    return 0;
                }
            }
        } else if (ret && strcmp(name_substr, "check_cnt") == 0) {
            builder->check_cnt = *(unsigned *)val;
            return 0;
        } else if (ret && strcmp(name_substr, "trigger") == 0) {
            builder->trigger = *(float *)val;
            return 0;
        } else if (ret && strcmp(name_substr, "degree") == 0) {
            builder->degree = *(float *)val;
            return 0;
        }
    }

    log_err("Invalid name: %s", query);
    return -1;
}

static struct memtier_memory *
builder_dynamic_create_memory(struct memtier_builder *builder)
{
    int i;
    struct memtier_memory *memory;
    struct memtier_threshold_cfg *thres;

    if (builder->cfg_size < 2) {
        log_err("There should be at least 2 tiers added to builder "
                "to use POLICY_DYNAMIC_THRESHOLD");
        return NULL;
    }
    thres = jemk_calloc(THRESHOLD_NUM(builder),
                        sizeof(struct memtier_threshold_cfg));
    if (!thres) {
        log_err("calloc() failed.");
        return NULL;
    }

    memory = memtier_memory_init(builder->cfg_size, true, false);
    if (!memory) {
        jemk_free(thres);
        log_err("memtier_memory_init failed.");
        return NULL;
    }

    memory->thres_init_check_cnt = builder->check_cnt;
    memory->thres_check_cnt = builder->check_cnt;
    memory->thres_trigger = builder->trigger;
    memory->thres_degree = builder->degree;

    memory->thres = thres;
    for (i = 0; i < THRESHOLD_NUM(builder); ++i) {
        memory->thres[i].val = builder->thres[i].val;
        memory->thres[i].min = builder->thres[i].min;
        memory->thres[i].max = builder->thres[i].max;
        memory->thres[i].exp_norm_ratio =
            builder->cfg[i + 1].kind_ratio / builder->cfg[i].kind_ratio;
    }

    // Validate threshold configuration:
    // * check if values of thresholds are in ascending order - each Nth
    //   threshold value has to be lower than (N+1)th value
    // * each threshold value has to be greater than min and lower than max
    //   value defined for this thresholds
    // * min/max ranges of adjacent threshold should not overlap - max
    //   value of Nth threshold has to be lower than min value of (N+1)th
    //   threshold
    // * threshold trigger and change values has to be positive values
    for (i = 0; i < THRESHOLD_NUM(builder); ++i) {
        if (memory->thres[i].min > memory->thres[i].val) {
            log_err("Minimum value of threshold %d "
                    "is too high (min = %zu, val = %zu)",
                    i, memory->thres[i].min, memory->thres[i].val);
            goto failure;
        } else if (memory->thres[i].val > memory->thres[i].max) {
            log_err("Maximum value of threshold %d "
                    "is too low (val = %zu, max = %zu)",
                    i, memory->thres[i].val, memory->thres[i].max);
            goto failure;
        }

        if ((i > 0) && (memory->thres[i - 1].max > memory->thres[i].min)) {
            log_err("Maximum value of threshold %d "
                    "should be less than minimum value of threshold %d",
                    i - 1, i);
            goto failure;
        }
    }

    if (memory->thres_degree < 0) {
        log_err("Threshold change value has to be >= 0");
        goto failure;
    }

    if (memory->thres_trigger < 0) {
        log_err("Threshold trigger value has to be >= 0");
        goto failure;
    }

    for (i = 1; i < memory->cfg_size; ++i) {
        memory->cfg[i].kind = builder->cfg[i].kind;
        memory->cfg[i].kind_ratio =
            builder->cfg[0].kind_ratio / builder->cfg[i].kind_ratio;
    }
    memory->cfg[0].kind = builder->cfg[0].kind;
    memory->cfg[0].kind_ratio = 1.0;

    return memory;

failure:
    jemk_free(memory->thres);
    jemk_free(memory->cfg);
    jemk_free(memory);
    return NULL;
}

extern pthread_t pebs_thread;

static struct memtier_memory *
builder_hot_create_memory(struct memtier_builder *builder)
{
    int i;
    // TODO use some properties? hotness weight should be configurable
    tachanka_init(OLD_TIME_WINDOW_HOTNESS_WEIGHT, RANKING_BUFFER_SIZE_ELEMENTS);
    pebs_init(getpid());

    struct memtier_memory *memory =
        memtier_memory_init(builder->cfg_size, false, true);
    memory->hot_tier_id = -1;

    if (memory->cfg_size != 2) {
        log_fatal("Incorrect number of tiers for data hotness policy");
        exit(-1);
    }

    double ratio_sum = builder->cfg[0].kind_ratio + builder->cfg[1].kind_ratio;

    // TODO requires cleanup
    for (i = 0; i < memory->cfg_size; ++i) {
        memory->cfg[i].kind = builder->cfg[i].kind;
        memory->cfg[i].kind_ratio =
            builder->cfg[i].kind_ratio / ratio_sum;
    // TODO - why this is commented out?
//         memory->cfg[i].kind_ratio =
//             builder->cfg[0].kind_ratio / builder->cfg[i].kind_ratio;
        if (memory->cfg[i].kind == MEMKIND_DEFAULT) {
            memory->hot_tier_id = i; // the usage of this variable might cause some confusion...
        }
    }

    // TODO requires cleanup
    memory->cfg[0].kind = builder->cfg[0].kind;
//     memory->cfg[0].kind_ratio = 1.0;
    if (memory->cfg[0].kind == MEMKIND_DEFAULT) {
        memory->hot_tier_id = 0;
    }

    if (memory->hot_tier_id == -1) {
        log_fatal("No tier suitable for HOT memory defined.");
        exit(-1);
    }
    double dram_total_ratio=memory->cfg[memory->hot_tier_id].kind_ratio;

#if PRINT_POLICY_CREATE_MEMORY_INFO
    struct timespec t;
    int ret = clock_gettime(CLOCK_MONOTONIC, &t);
    if (ret != 0) {
        log_fatal("Create Memory: ASSERT CREATE FAILURE!\n");
    }

    log_info("creates memory [ratio %f], timespec [seconds, nanoseconds]: [%ld, %ld]",
        dram_total_ratio, t.tv_sec, t.tv_nsec);
#endif

    tachanka_set_dram_total_ratio(dram_total_ratio);
    return memory;
}

static int builder_dynamic_update(struct memtier_builder *builder)
{
    if (builder->cfg_size < 1)
        return 0;

    struct memtier_threshold_cfg *thres =
        jemk_realloc(builder->thres, sizeof(*thres) * (builder->cfg_size + 1));
    if (!thres) {
        log_err("realloc() failed.");
        return -1;
    }
    builder->thres = thres;
    int th_indx = builder->cfg_size - 1;
    builder->thres[th_indx].min = THRESHOLD_STEP * (0.5 + th_indx);
    builder->thres[th_indx].val = THRESHOLD_STEP * builder->cfg_size;
    builder->thres[th_indx].max = THRESHOLD_STEP * (1.5 + th_indx) - 1;

    return 0;
}

static memtier_policy_t pol = 0; // FIXME: not a global

MEMKIND_EXPORT struct memtier_builder *
memtier_builder_new(memtier_policy_t policy)
{
    struct memtier_builder *b = jemk_calloc(1, sizeof(struct memtier_builder));
    if (b) {
        pol = policy;
        //printf("policy %d\n", pol);
        switch (policy) {
            case MEMTIER_POLICY_STATIC_RATIO:
                b->create_mem = builder_static_create_memory;
                b->update_builder = NULL;
                b->ctl_set = builder_static_ctl_set;
                b->cfg = NULL;
                b->thres = NULL;
                return b;
            case MEMTIER_POLICY_DYNAMIC_THRESHOLD:
                b->create_mem = builder_dynamic_create_memory;
                b->update_builder = builder_dynamic_update;
                b->ctl_set = builder_dynamic_ctl_set;
                b->cfg = NULL;
                b->thres = NULL;
                b->check_cnt = THRESHOLD_CHECK_CNT;
                b->trigger = THRESHOLD_TRIGGER;
                b->degree = THRESHOLD_DEGREE;
                return b;
            case MEMTIER_POLICY_DATA_HOTNESS:
                b->create_mem = builder_hot_create_memory;
                b->update_builder = NULL;
                b->ctl_set = builder_hot_ctl_set;
                b->cfg = NULL;
                b->thres = NULL;
                return b;
            default:
                log_err("Unrecognized memory policy %u", policy);
                jemk_free(b);
        }
    }
    return NULL;
}

MEMKIND_EXPORT void memtier_builder_delete(struct memtier_builder *builder)
{
    print_builder(builder);
    jemk_free(builder->thres);
    jemk_free(builder->cfg);
    jemk_free(builder);
}

MEMKIND_EXPORT int memtier_builder_add_tier(struct memtier_builder *builder,
                                            memkind_t kind, unsigned kind_ratio)
{
    int i;

    if (!kind) {
        log_err("Kind is empty.");
        return -1;
    }

    for (i = 0; i < builder->cfg_size; ++i) {
        if (kind == builder->cfg[i].kind) {
            log_err("Kind is already in builder.");
            return -1;
        }
    }

    if (builder->update_builder) {
        if (builder->update_builder(builder)) {
            log_err("Update builder operation failed");
            return -1;
        }
    }

    struct memtier_tier_cfg *cfg =
        jemk_realloc(builder->cfg, sizeof(*cfg) * (builder->cfg_size + 1));

    if (!cfg) {
        log_err("realloc() failed.");
        return -1;
    }

    builder->cfg = cfg;
    builder->cfg[builder->cfg_size].kind = kind;
    builder->cfg[builder->cfg_size].kind_ratio = kind_ratio;
    builder->cfg_size += 1;
    return 0;
}

MEMKIND_EXPORT struct memtier_memory *
memtier_builder_construct_memtier_memory(struct memtier_builder *builder)
{
#if PRINT_POLICY_CONSTRUCT_MEMORY_INFO
    struct timespec t;
    int ret = clock_gettime(CLOCK_MONOTONIC, &t);
    if (ret != 0) {
        log_fatal("ASSERT CONSTRUCT FAILURE!");
        exit(-1);
    }

    log_info("constructs memory, timespec [seconds, nanoseconds]: [%ld, %ld]",
        t.tv_sec, t.tv_nsec);
#endif

    return builder->create_mem(builder);
}

MEMKIND_EXPORT void memtier_delete_memtier_memory(struct memtier_memory *memory)
{
    pebs_fini(); // TODO conditional - only if pebs started

#if PRINT_POLICY_DELETE_MEMORY_INFO
    struct timespec t;
    int ret = clock_gettime(CLOCK_MONOTONIC, &t);
        if (ret != 0) {
        log_fatal("Delete Memory: ASSERT DELETE FAILURE!");
        exit(-1);
    }

    log_info("delete memory, timespec [seconds, nanoseconds]: [%ld, %ld]",
        t.tv_sec, t.tv_nsec);
#endif

    print_memtier_memory(memory);
    jemk_free(memory->thres);
    jemk_free(memory->cfg);
    jemk_free(memory);
}

// TODO - create "get" version for builder
// TODO - create "get" version for memtier_memory obj (this will be read-only)
// TODO - how to validate val type? e.g. provide function with explicit size_t
//        type of val for thresholds[ID].val/min/max
MEMKIND_EXPORT int memtier_ctl_set(struct memtier_builder *builder,
                                   const char *name, const void *val)
{
    return builder->ctl_set(builder, name, val);
}

MEMKIND_EXPORT void *memtier_malloc(struct memtier_memory *memory, size_t size)
{
    void *ptr;
    uint64_t data;

    ptr = memtier_kind_malloc(memory->get_kind(memory, size, &data), size);
    memory->post_alloc(data, ptr, size);
    memory->update_cfg(memory);

    return ptr;
}

MEMKIND_EXPORT void *memtier_kind_malloc(memkind_t kind, size_t size)
{
//     static atomic_uint_fast16_t counter=0;
//     const uint64_t interval=1000;
//     if (++counter > interval) {
//         struct timespec t;
//         int ret = clock_gettime(CLOCK_MONOTONIC, &t);
//         if (ret != 0) {
//             printf("ASSERT MEMTIER MALLOC COUNTER FAILURE!\n");
//         }
//         assert(ret == 0);
//         printf("malloc counter %lu hit, [seconds, nanoseconds]: [%ld, %ld]\n",
//             interval, t.tv_sec, t.tv_nsec);
//         counter=0u;
//     }

    void *ptr = memkind_malloc(kind, size);
    increment_alloc_size(kind->partition, jemk_malloc_usable_size(ptr));
#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_malloc_post)
        memtier_kind_malloc_post(kind, size, &ptr);
#endif
    return ptr;
}

MEMKIND_EXPORT void *memtier_calloc(struct memtier_memory *memory, size_t num,
                                    size_t size)
{
//     static atomic_uint_fast16_t counter=0;
//     const uint64_t interval=100;
//     if (++counter > interval) {
//         struct timespec t;
//         int ret = clock_gettime(CLOCK_MONOTONIC, &t);
//         if (ret != 0) {
//             printf("ASSERT MEMTIER CALLOC COUNTER FAILURE!\n");
//         }
//         assert(ret == 0);
//         printf("calloc counter %lu hit, [seconds, nanoseconds]: [%ld, %ld]\n",
//             interval, t.tv_sec, t.tv_nsec);
//         counter=0u;
//     }
    void *ptr;
    uint64_t data;

    ptr = memtier_kind_calloc(memory->get_kind(memory, size, &data), num, size);
    memory->post_alloc(data, ptr, size);
    memory->update_cfg(memory);

    return ptr;
}

MEMKIND_EXPORT void *memtier_kind_calloc(memkind_t kind, size_t num,
                                         size_t size)
{
    void *ptr = memkind_calloc(kind, num, size);
    increment_alloc_size(kind->partition, jemk_malloc_usable_size(ptr));

#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_calloc_post)
        memtier_kind_calloc_post(kind, num, size, &ptr);
#endif
    return ptr;
}

MEMKIND_EXPORT void *memtier_realloc(struct memtier_memory *memory, void *ptr,
                                     size_t size)
{
    // reallocate inside same kind
    if (ptr) {
        struct memkind *kind = memkind_detect_kind(ptr);
        ptr = memtier_kind_realloc(kind, ptr, size);
        memory->update_cfg(memory);

        return ptr;
    }

    return memtier_malloc(memory, size);
}

MEMKIND_EXPORT void *memtier_kind_realloc(memkind_t kind, void *ptr,
                                          size_t size)
{
    if (size == 0 && ptr != NULL) {
#ifdef MEMKIND_DECORATION_ENABLED
        if (memtier_kind_free_pre)
            memtier_kind_free_pre(&ptr);
#endif

        if (pol == MEMTIER_POLICY_DATA_HOTNESS) {
//             unregister_block(ptr);
            EventEntry_t entry = {
                .type = EVENT_DESTROY_REMOVE,
                .data.destroyRemoveData = {
                .address = ptr,
                }
            };

            bool success = tachanka_ranking_event_push(&entry);
#if PRINT_POLICY_LOG_STATISTICS_INFO
            if (success) {
                g_successful_adds++;
                g_successful_adds_realloc0++;
            } else {
                g_failed_adds++;
                g_failed_adds_realloc0++;
            }
#else
            (void)success;
#endif
        }
        decrement_alloc_size(kind->partition, jemk_malloc_usable_size(ptr));
        memkind_free(kind, ptr);
        return NULL;
    } else if (ptr == NULL) {
        return memtier_kind_malloc(kind, size);
    }
    decrement_alloc_size(kind->partition, jemk_malloc_usable_size(ptr));

    void *n_ptr = memkind_realloc(kind, ptr, size);
    if (pol == MEMTIER_POLICY_DATA_HOTNESS) {
        // TODO offload to separate thread
        EventEntry_t entry = {
            .type = EVENT_REALLOC,
            .data.reallocData = {
                .addressOld = ptr,
                .addressNew = n_ptr,
                .size = size,
            }
        };
        bool success = tachanka_ranking_event_push(&entry);
#if PRINT_POLICY_LOG_STATISTICS_INFO
        if (success) {
            g_successful_adds++;
            g_successful_adds_realloc1++;
        } else {
            g_failed_adds++;
            g_failed_adds_realloc1++;
        }
#else
        (void)success;
#endif
//         realloc_block(ptr, n_ptr, size);
    }
    increment_alloc_size(kind->partition, jemk_malloc_usable_size(n_ptr));
#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_realloc_post)
        memtier_kind_realloc_post(kind, ptr, size, &n_ptr);
#endif
    return n_ptr;
}

MEMKIND_EXPORT int memtier_posix_memalign(struct memtier_memory *memory,
                                          void **memptr, size_t alignment,
                                          size_t size)
{
    uint64_t data = 0;
    int ret = memtier_kind_posix_memalign(memory->get_kind(memory, size, &data),
                                          memptr, alignment, size);
    memory->post_alloc(data, *memptr, size);
    memory->update_cfg(memory);

    return ret;
}

MEMKIND_EXPORT int memtier_kind_posix_memalign(memkind_t kind, void **memptr,
                                               size_t alignment, size_t size)
{
    // TODO: hotness
    int res = memkind_posix_memalign(kind, memptr, alignment, size);
    increment_alloc_size(kind->partition, jemk_malloc_usable_size(*memptr));
#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_posix_memalign_post)
        memtier_kind_posix_memalign_post(kind, memptr, alignment, size, &res);
#endif
    return res;
}

MEMKIND_EXPORT size_t memtier_usable_size(void *ptr)
{
    size_t size = jemk_malloc_usable_size(ptr);
#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_usable_size_post)
        memtier_kind_usable_size_post(&ptr, size);
#endif
    return size;
}

MEMKIND_EXPORT void memtier_kind_free(memkind_t kind, void *ptr)
{
#ifdef MEMKIND_DECORATION_ENABLED
    if (memtier_kind_free_pre)
        memtier_kind_free_pre(&ptr);
#endif
    if (!kind) {
        kind = memkind_detect_kind(ptr);
        if (!kind)
            return;
    }

    if (pol == MEMTIER_POLICY_DATA_HOTNESS) {
        // TODO offload to PEBS (ranking_queue) !!! Currently contains race conditions
//         unregister_block(ptr);

        EventEntry_t entry = {
            .type = EVENT_DESTROY_REMOVE,
            .data.destroyRemoveData = {
                .address = ptr,
            }
        };
        bool success = tachanka_ranking_event_push(&entry);
#if PRINT_POLICY_LOG_STATISTICS_INFO
        if (success) {
            g_successful_adds++;
            g_successful_adds_free++;
        } else {
            g_failed_adds++;
            g_failed_adds_free++;
        }
#else
        (void)success;
#endif
    }
    decrement_alloc_size(kind->partition, jemk_malloc_usable_size(ptr));
    memkind_free(kind, ptr);
}

MEMKIND_EXPORT size_t memtier_kind_allocated_size(memkind_t kind)
{
    size_t size_ret;
    long long size_all = 0;
    long long size;
    unsigned bucket_id;

    for (bucket_id = 0; bucket_id < THREAD_BUCKETS; ++bucket_id) {
        size = memkind_atomic_get_and_zeroing(
            t_alloc_size[kind->partition][bucket_id]);
        size_all += size;
    }
    size_ret =
        memkind_atomic_increment(g_alloc_size[kind->partition], size_all);
    return (size_ret + size_all);
}
