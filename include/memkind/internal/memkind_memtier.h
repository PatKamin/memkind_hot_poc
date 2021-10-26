// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2021 Intel Corporation. */

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <memkind.h>
#include <stdlib.h>

/**
 * Header file for the memkind heap manager.
 * More details in memtier(3) man page.
 *
 * API standards are described in memtier(3) man page.
 */

/// \brief Forward declaration
struct memtier_builder;
struct memtier_memory;

typedef enum memtier_policy_t
{
    /**
     * Static Ratio policy
     */
    MEMTIER_POLICY_STATIC_RATIO = 0,

    /**
     * Dynamic Threshold policy
     */
    MEMTIER_POLICY_DYNAMIC_THRESHOLD = 1,

    /**
     * Hotness prediction policy
     */
    MEMTIER_POLICY_DATA_HOTNESS = 2,

    /**
     * Max policy value.
     */
    MEMTIER_POLICY_MAX_VALUE
} memtier_policy_t;

///
/// \brief Create a memtier builder
/// \note STANDARD API
/// \param policy memtier policy
/// \return memtier builder, NULL on failure
///
struct memtier_builder *memtier_builder_new(memtier_policy_t policy);

///
/// \brief Delete memtier builder
/// \note STANDARD API
/// \param builder memtier builder
///
void memtier_builder_delete(struct memtier_builder *builder);

///
/// \brief Add memtier tier to memtier builder
/// \note STANDARD API
/// \param builder memtier builder
/// \param kind memkind kind
/// \param kind_ratio expected memtier tier ratio
/// \return Operation status, 0 on success, other values on
/// failure
///
int memtier_builder_add_tier(struct memtier_builder *builder, memkind_t kind,
                             unsigned kind_ratio);

///
/// \brief Construct a memtier memory
/// \note STANDARD API
/// \param builder memtier builder
/// \return Pointer to constructed memtier memory object
///
struct memtier_memory *
memtier_builder_construct_memtier_memory(struct memtier_builder *builder);

///
/// \brief Delete memtier memory
/// \note STANDARD API
/// \param memory memtier memory
///
void memtier_delete_memtier_memory(struct memtier_memory *memory);

///
/// \brief Allocates size bytes of uninitialized storage of the specified
///        memtier memory
/// \note STANDARD API
/// \param memory specified memtier memory
/// \param size number of bytes to allocate
/// \return Pointer to the allocated memory
///
void *memtier_malloc(struct memtier_memory *memory, size_t size);

///
/// \brief Allocates size bytes of uninitialized storage of the specified kind
/// \note STANDARD API
/// \param kind specified memkind kind
/// \param size number of bytes to allocate
/// \return Pointer to the allocated memory
void *memtier_kind_malloc(memkind_t kind, size_t size);

///
/// \brief Allocates memory of the specified memtier memory for an array of num
///        elements of size bytes each and initializes all bytes in the
///        allocated storage to zero
/// \note STANDARD API
/// \param memory specified memtier memory
/// \param num number of objects
/// \param size specified size of each element
/// \return Pointer to the allocated memory
///
void *memtier_calloc(struct memtier_memory *memory, size_t num, size_t size);

///
/// \brief Allocates memory of the specified kind for an array of num
///        elements of size bytes each and initializes all bytes in the
///        allocated storage to zero
/// \note STANDARD API
/// \param kind specified memkind kind
/// \param num number of objects
/// \param size specified size of each element
/// \return Pointer to the allocated memory
///
void *memtier_kind_calloc(memkind_t kind, size_t num, size_t size);

///
/// \brief Reallocates memory of the specified memtier memory
/// \note STANDARD API
/// \param memory specified memtier memory
/// \param ptr pointer to the memory block to be reallocated
/// \param size new size for the memory block in bytes
/// \return Pointer to the allocated memory
///
void *memtier_realloc(struct memtier_memory *memory, void *ptr, size_t size);

///
/// \brief Reallocates memory of the specified kind
/// \note STANDARD API
/// \param kind specified memkind kind
/// \param ptr pointer to the memory block to be reallocated
/// \param size new size for the memory block in bytes
/// \return Pointer to the allocated memory
///
void *memtier_kind_realloc(memkind_t kind, void *ptr, size_t size);

///
/// \brief Allocates size bytes of the specified memtier memory and places the
///        address of the allocated memory in *memptr. The address of the
//         allocated memory will be a multiple of alignment, which must be a
///        power of two and a multiple of sizeof(void*)
/// \note STANDARD API
/// \param memory specified memtier memory
/// \param memptr address of the allocated memory
/// \param alignment specified alignment of bytes
/// \param size specified size of bytes
/// \return operation status, 0 on success, EINVAL or
///         ENOMEM on failure
///
int memtier_posix_memalign(struct memtier_memory *memory, void **memptr,
                           size_t alignment, size_t size);

///
/// \brief Allocates size bytes of the specified kind and places the
///        address of the allocated memory in *memptr. The address of the
///        allocated memory will be a multiple of alignment, which must be a
///        power of two and a multiple of sizeof(void*)
/// \note STANDARD API
/// \param kind specified memkind kind
/// \param memptr address of the allocated memory
/// \param alignment specified alignment of bytes
/// \param size specified size of bytes
/// \return operation status, 0 on success, EINVAL or
///         ENOMEM on failure
///
int memtier_kind_posix_memalign(memkind_t kind, void **memptr, size_t alignment,
                                size_t size);

///
/// \brief Obtain size of block of memory allocated with the memtier API
/// \note STANDARD API
/// \param ptr pointer to the allocated memory
/// \return Number of usable bytes
///
size_t memtier_usable_size(void *ptr);

///
/// \brief Free the memory space allocated with the memtier_kind API
/// \note STANDARD API
/// \param kind specified memkind kind
/// \param ptr pointer to the allocated memory
///
void memtier_kind_free(memkind_t kind, void *ptr);

///
/// \brief Free the memory space allocated with the memtier API
/// \note STANDARD API
/// \param ptr pointer to the allocated memory
///
static inline void memtier_free(void *ptr)
{
    memtier_kind_free(NULL, ptr);
}

///
/// \brief Obtain size of allocated memory with the memtier API inside
///        specified kind
/// \note STANDARD API
/// \param kind specified memkind kind
/// \return Number of usable bytes
///
size_t memtier_kind_allocated_size(memkind_t kind);

///
/// \brief Set memtier property
/// \note STANDARD API
/// \param builder memtier builder
/// \param name name of the property
/// \param val value to set
/// \return Operation status, 0 on success, other values on
/// failure
///
int memtier_ctl_set(struct memtier_builder *builder, const char *name,
                    const void *val);


double memtier_kind_get_actual_hot_to_total_allocated_ratio(void);
double memtier_kind_get_actual_hot_to_total_desired_ratio(void);

// DEBUG
// float get_obj_hotness(int size);

// PEBS
extern double sample_frequency;
extern double pebs_freq_hz;
#define MMAP_DATA_SIZE   8

// critnib
// #define INIT_MALLOC_HOTNESS   20u
#define INIT_MALLOC_HOTNESS 1u // TODO this does not work, at least for now
#define MAXTYPES            1*1024*1024
#define MAXBLOCKS           16*1024*1024

// bthash
#define CUSTOM_BACKTRACE 1
#define STACK_RANGE 1
#define REDUCED_STACK_SEARCH 1
#define SIMD_INSTRUCTIONS 0
#define LIB_BINSEARCH 0
#define FINALIZE_HASH 0

// hotness calculation
extern unsigned long long hotness_measure_window;
extern double old_time_window_hotness_weight;
#define RANKING_BUFFER_SIZE_ELEMENTS    1000000 // TODO make tests, add error handling and come up with some sensible value
#define RANKING_TOUCH_ALL 1

// logging
#define PEBS_LOG_TO_FILE 0

#define PRINT_PEBS_BASIC_INFO 1
#define PRINT_PEBS_TIMESPEC_DEADLINE_INFO 0
#define PRINT_PEBS_STATS_ON_COUNTER_OVERFLOW_INFO 0
#define PRINT_PEBS_NEW_DATA_INFO 0
#define PRINT_PEBS_TOUCH_INFO 0
#define PRINT_PEBS_SAMPLES_NUM_INFO 1

#define PRINT_CRITNIB_NEW_BLOCK_REGISTERED_INFO 0
#define PRINT_CRITNIB_UNREGISTER_BLOCK_INFO 0
#define PRINT_CRITNIB_TOUCH_INFO 0
#define PRINT_CRITNIB_REALLOC_INFO 0
#define PRINT_CRITNIB_NOT_FOUND_ON_TOUCH_WARNING 0
#define PRINT_CRITNIB_NOT_FOUND_ON_UNREGISTER_BLOCK_WARNING 0
#define PRINT_CRITNIB_NOT_FOUND_ON_REALLOC_WARNING 0

#define PRINT_POLICY_LOG_STATISTICS_INFO 1
#define CRASH_ON_BLOCK_NOT_FOUND 0
#define PRINT_POLICY_BACKTRACE_INFO 0
#define PRINT_POLICY_CREATE_MEMORY_INFO 1
#define PRINT_POLICY_CONSTRUCT_MEMORY_INFO 0
#define PRINT_POLICY_DELETE_MEMORY_INFO 0

#define CHECK_ADDED_SIZE 0

#define QUANTIFICATION_ENABLED 0
#define RANKING_FIXER_ENABLED 1
#define INTERPOLATED_THRESH 0
#define FALLBACK_TO_STATIC 0

// when buffer is full, waits until it can re-add elements
// this feature can negativly impact performance!
#define ASSURE_RANKING_DELIVERY 0
#define OFFLOAD_RANKING_OPS_TO_BACKGROUD_THREAD 1

#if QUANTIFICATION_ENABLED
typedef int quantified_hotness_t;
#else
typedef double quantified_hotness_t;
#endif

#ifdef __cplusplus
}
#endif
