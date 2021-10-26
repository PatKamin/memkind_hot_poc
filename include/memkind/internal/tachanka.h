#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"
#include "stdlib.h"
// #include "stdatomic.h" issues with include - this is "extern" code from c++
#include "asm-generic/int-ll64.h"
#include "ranking_queue.h"

typedef enum TimestampState {
    TIMESTAMP_NOT_SET,
    TIMESTAMP_INIT,
    TIMESTAMP_INIT_DONE,
} TimestampState_t;

typedef enum Hotness {
    HOTNESS_HOT=0,
    HOTNESS_COLD=1,
    HOTNESS_NOT_FOUND=2,
} Hotness_e;

typedef void (*tachanka_touch_callback)(void*);

void register_block(uint64_t hash, void *addr, size_t size);
void register_block_in_ranking(void *addr, size_t size);
void unregister_block(void *addr);
void realloc_block(void *addr, void *new_addr, size_t size);
void *new_block(size_t size);
void touch(void *addr, __u64 timestamp, int from_malloc);
void tachanka_init(double old_window_hotness_weight, size_t event_queue_size);
void tachanka_destroy(void);
void tachanka_update_threshold(void);
void tachanka_set_dram_total_ratio(double desired, double actual);
double tachanka_get_obj_hotness(int size);
double tachanka_get_addr_hotness(void *addr);
// double tachanka_set_touch_callback(void *addr, const char*name);
int tachanka_set_touch_callback(void *addr, tachanka_touch_callback cb, void* arg);
Hotness_e tachanka_get_hotness_type(const void *addr);
Hotness_e tachanka_get_hotness_type_hash(uint64_t hash);
double tachanka_get_hot_thresh(void);
bool tachanka_ranking_event_push(EventEntry_t *event);
bool tachanka_ranking_event_pop(EventEntry_t *event);

/// \brief Touch every ttype object to update hotness
/// \param timestamp timestamp from pebs
/// \param add_hotness hotness value to be added to all types
void tachanka_ranking_touch_all(__u64 timestamp, double add_hotness);

/// \brief Getter for type's frequency
/// \param index index of a type in ttypes list 
double tachanka_get_frequency(size_t index);

/// \brief Getter for type's timestamp state
/// \param index index of a type in ttypes list
TimestampState_t tachanka_get_timestamp_state(size_t index);

struct ttype {
    uint64_t hash;
    int num_allocs; // TODO
    int total_size; // TODO

    __u64 t2;   // start of previous measurement window
    __u64 t1;   // start of current window
    __u64 t0;   // timestamp of last processed data

    double n2;   // add hotness in prev window
    double n1;   // add hotness in current window

    tachanka_touch_callback touchCb;
    void *touchCbArg;
    // TODO f should be atomic, but there are issues with includes
    // worst thing that can happen without atomicity:
    // incorrect hot/cold classification (read without a mutex in ranking_is_hot)
    double f;  // frequency - current
    TimestampState_t timestamp_state;
};

struct tblock
{
    void *addr;
    size_t size;
    int type;
    int nextfree; // can reuse one of other fields
};

#ifdef __cplusplus
}
#endif
