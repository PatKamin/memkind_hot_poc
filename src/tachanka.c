#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <memkind/internal/pebs.h>
#include <memkind/internal/bthash.h>
#include <memkind/internal/critnib.h>
#include <memkind/internal/tachanka.h>

#define MAXBLOCKS 16*1048576
struct tblock tblocks[MAXBLOCKS];

static int nblocks = 0;

/*static*/ critnib *hash_to_block, *addr_to_block;

int is_hot(uint64_t hash)
{
    return 1;
}

#define HOTNESS_MEASURE_WINDOW 1000000000ULL
#define MALLOC_HOTNESS 20

void register_block(uint64_t hash, void *addr, size_t size)
{
    struct tblock *bl = &tblocks[__sync_fetch_and_add(&nblocks, 1)];
    if (bl >= &tblocks[MAXBLOCKS])
        fprintf(stderr, "Too many allocated blocks\n"), exit(1);

    bl->addr = addr;
    bl->size = size;
    bl->hot_or_not = -2; // no time set

    struct tblock *pbl = critnib_get(hash_to_block, hash);
    if (pbl) {
        bl->parent = pbl - tblocks;
        pbl->num_allocs++;
        pbl->total_size += size;
    }
    else {
        bl->parent = -1;
        critnib_insert(hash_to_block, hash, bl, 0);
    }

    critnib_insert(addr_to_block, (uintptr_t)addr, bl, 0);
}

void touch(void *addr, __u64 timestamp, int from_malloc)
{
    struct tblock *bl = critnib_find_le(addr_to_block, (uintptr_t)addr);
    if (!bl)
        return;
    if (addr >= bl->addr + bl->size)
        return;
    if (bl->parent != -1)
        bl = &tblocks[bl->parent];

    // TODO - is this thread safeness needed? or best effort will be enough?
    //__sync_fetch_and_add(&bl->accesses, 1);

    if (from_malloc) {
        bl->n2 += MALLOC_HOTNESS;
    } else {
        bl->t0 = timestamp;
        if (bl->hot_or_not == -2) {
            bl->t2 = timestamp;
            bl->hot_or_not = -1;
        }
    }

    // check if type is ready for classification
    if (bl->hot_or_not < 0) {
        bl->n2 ++; // TODO - not thread safe is ok?

        // check if data is measured for time enough to classify hotness
        if ((bl->t0 - bl->t2) > HOTNESS_MEASURE_WINDOW) {
            // TODO - classify hotness
            bl->hot_or_not = 1;
            bl->t1 = bl->t0;
        }
    } else {
        bl->n1 ++; // TODO - not thread safe is ok?
        if ((bl->t0 - bl->t1) > HOTNESS_MEASURE_WINDOW) {
            // move to next measurement window
            float f2 = (float)bl->n2 * bl->t2 / (bl->t2 - bl->t0);
            float f1 = (float)bl->n1 * bl->t1 / (bl->t2 - bl->t0);
            bl->f = f2 * 0.3 + f1 * 0.7; // TODO weighted sum or sth else?
            bl->t2 = bl->t1;
            bl->t1 = bl->t0;
            bl->n2 = bl->n1;
            bl->n1 = 0;
        }
    }
}

void tachanka_init(void)
{
    read_maps();
    hash_to_block = critnib_new();
    addr_to_block = critnib_new();
}


// DEBUG

#ifndef MEMKIND_EXPORT
#define MEMKIND_EXPORT __attribute__((visibility("default")))
#endif

MEMKIND_EXPORT float get_obj_hotness(int size)
{
    for (int i = 0; i < 20; i++) {
        struct tblock* tb = (struct tblock*)critnib_get_leaf(hash_to_block, i);

        if (tb != NULL && tb->size == size) {
             return tb->f;
        }
    }

    return -1;
}
